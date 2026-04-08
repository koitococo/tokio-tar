use crate::fs::normalize;
use crate::{
    error::TarError,
    fs::{normalize_absolute, normalize_relative},
    header::bytes2path,
    other,
    pax::pax_extensions,
    Archive, Header, PaxExtensions,
};
use filetime::{self, FileTime};
use rustc_hash::FxHashSet;
use std::{
    borrow::Cow,
    cmp,
    collections::VecDeque,
    convert::TryFrom,
    fmt,
    io::{Error, ErrorKind, SeekFrom},
    marker,
    path::{Component, Path, PathBuf},
    pin::Pin,
    task::{Context, Poll},
};
use tokio::{
    fs,
    fs::{remove_file, OpenOptions},
    io::{self, AsyncRead as Read, AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
};

/// A read-only view into an entry of an archive.
///
/// This structure is a window into a portion of a borrowed archive which can
/// be inspected. It acts as a file handle by implementing the Reader trait. An
/// entry cannot be rewritten once inserted into an archive.
pub struct Entry<R: Read + Unpin> {
    fields: EntryFields<R>,
    _ignored: marker::PhantomData<Archive<R>>,
}

impl<R: Read + Unpin> fmt::Debug for Entry<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry")
            .field("fields", &self.fields)
            .finish()
    }
}

// private implementation detail of `Entry`, but concrete (no type parameters)
// and also all-public to be constructed from other modules.
pub struct EntryFields<R: Read + Unpin> {
    pub long_pathname: Option<Vec<u8>>,
    pub long_linkname: Option<Vec<u8>>,
    pub pax_extensions: Option<Vec<u8>>,
    pub header: Header,
    pub size: u64,
    pub header_pos: u64,
    pub file_pos: u64,
    pub data: VecDeque<EntryIo<R>>,
    pub unpack_xattrs: bool,
    pub preserve_permissions: bool,
    pub preserve_ownerships: bool,
    pub preserve_mtime: bool,
    pub overwrite: bool,
    pub allow_external_symlinks: bool,
    pub(crate) read_state: Option<EntryIo<R>>,
}

impl<R: Read + Unpin> fmt::Debug for EntryFields<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EntryFields")
            .field("long_pathname", &self.long_pathname)
            .field("long_linkname", &self.long_linkname)
            .field("pax_extensions", &self.pax_extensions)
            .field("header", &self.header)
            .field("size", &self.size)
            .field("header_pos", &self.header_pos)
            .field("file_pos", &self.file_pos)
            .field("data", &self.data)
            .field("unpack_xattrs", &self.unpack_xattrs)
            .field("preserve_permissions", &self.preserve_permissions)
            .field("preserve_mtime", &self.preserve_mtime)
            .field("overwrite", &self.overwrite)
            .field("allow_external_symlinks", &self.allow_external_symlinks)
            .field("read_state", &self.read_state)
            .finish()
    }
}

pub enum EntryIo<R: Read + Unpin> {
    Pad(io::Take<io::Repeat>),
    Data(io::Take<R>),
}

impl<R: Read + Unpin> fmt::Debug for EntryIo<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntryIo::Pad(t) => write!(f, "EntryIo::Pad({})", t.limit()),
            EntryIo::Data(t) => write!(f, "EntryIo::Data({})", t.limit()),
        }
    }
}

impl<R: Read + Unpin> Entry<R> {
    /// Returns the path name for this entry.
    ///
    /// Note that this function will convert any `\` characters to directory
    /// separators, and it will not always return the same value as
    /// `self.header().path()` as some archive formats have support for longer
    /// path names described in separate entries.
    ///
    /// It is recommended to use this method instead of inspecting the `header`
    /// directly to ensure that various archive formats are handled correctly.
    ///
    /// # Security Considerations
    ///
    /// The returned path is not normalized. On filesystems with complex behaviors
    /// (like Unicode normalization on APFS/HFS+ or case folding on Windows/macOS),
    /// distinct byte sequences may resolve to the same file.
    ///
    /// See the "Security Considerations" section in the crate [README] for details on mitigating these risks.
    ///
    /// [README]: https://github.com/astral-sh/tokio-tar#security-considerations
    pub fn path(&self) -> io::Result<Cow<'_, Path>> {
        self.fields.path()
    }

    /// Returns the raw bytes listed for this entry.
    ///
    /// Note that this function will convert any `\` characters to directory
    /// separators, and it will not always return the same value as
    /// `self.header().path_bytes()` as some archive formats have support for
    /// longer path names described in separate entries.
    ///
    /// This method may return an error if PAX extensions are malformed.
    pub fn path_bytes(&self) -> io::Result<Cow<'_, [u8]>> {
        self.fields.path_bytes()
    }

    /// Returns the link name for this entry, if any is found.
    pub fn link_name(&self) -> io::Result<Option<Cow<'_, Path>>> {
        self.fields.link_name()
    }

    /// Returns the raw link name for this entry, if any is found.
    pub fn link_name_bytes(&self) -> io::Result<Option<Cow<'_, [u8]>>> {
        self.fields.link_name_bytes()
    }

    /// Returns an iterator over the pax extensions contained in this entry.
    ///
    /// Pax extensions are a form of archive where extra metadata is stored in
    /// key/value pairs in entries before the entry they're intended to
    /// describe. For example this can be used to describe long file name or
    /// other metadata like atime/ctime/mtime in more precision.
    ///
    /// The returned iterator will yield key/value pairs for each extension.
    ///
    /// `None` will be returned if this entry does not indicate that it itself
    /// contains extensions, or if there were no previous extensions describing
    /// it.
    ///
    /// Note that global pax extensions are intended to be applied to all
    /// archive entries.
    ///
    /// Also note that this function will read the entire entry if the entry
    /// itself is a list of extensions.
    pub async fn pax_extensions(&mut self) -> io::Result<Option<PaxExtensions<'_>>> {
        self.fields.pax_extensions().await
    }

    /// Returns access to the header of this entry in the archive.
    ///
    /// This provides access to the metadata of the entry, including its name,
    /// permissions, size, and other metadata.
    pub fn header(&self) -> &Header {
        &self.fields.header
    }

    /// Returns the size of the data in this entry.
    pub fn size(&self) -> u64 {
        self.fields.size
    }

    /// Returns whether the entry is a directory.
    pub fn is_dir(&self) -> bool {
        self.header().entry_type().is_dir()
    }

    /// Returns whether the entry is a symlink.
    pub fn is_symlink(&self) -> bool {
        self.header().entry_type().is_symlink()
    }

    /// Returns whether the entry is a hard link.
    pub fn is_hard_link(&self) -> bool {
        self.header().entry_type().is_hard_link()
    }

    /// Unpacks this entry into the specified destination.
    ///
    /// This function is provided as a convenience for easily extracting the
    /// contents of one entry into a particular destination. It is equivalent to
    /// calling `EntryFields::unpack` on the underlying fields.
    ///
    /// This function will determine what kind of file is pointed to by this
    /// entry and create it at the location `dst` with appropriate permissions
    /// if possible.
    ///
    /// If `dst` is a relative path, it is treated as relative to the current
    /// working directory.
    ///
    /// # Errors
    ///
    /// This function will return an error if it is unable to create the file at
    /// the destination, if it fails to write the contents of the file, or if
    /// the entry is not a regular file, directory, or symlink.
    ///
    /// If the entry is a hard link, it will also return an error if the link
    /// target does not exist.
    pub async fn unpack<P: AsRef<Path>>(&mut self, dst: P) -> io::Result<()> {
        self.fields.unpack(dst).await
    }

    /// Unpacks this entry into the specified destination with additional safety checks.
    ///
    /// This function is similar to `unpack`, but performs additional safety
    /// checks to ensure that the entry will not be unpacked outside of the
    /// destination directory.
    ///
    /// If `dst` is not an absolute path, it will be treated as relative to the
    /// current working directory.
    ///
    /// See the "Security Considerations" section in the crate [README] for details.
    ///
    /// [README]: https://github.com/astral-sh/tokio-tar#security-considerations
    pub async fn unpack_in<P: AsRef<Path>>(&mut self, dst: P) -> io::Result<()> {
        self.fields.unpack_in(dst).await
    }

    /// Unpacks this entry into the specified destination with memoized validation.
    ///
    /// This is like `unpack_in`, but accepts pre-computed information from a previous
    /// call to `unpack_in` or `unpack_in_raw` to avoid redundant filesystem operations.
    ///
    /// The caller is responsible for ensuring that `dst` is the same canonical path
    /// passed to the previous call, and that `validated_paths` is the set of paths
    /// that have already been validated.
    pub async fn unpack_in_raw<P: AsRef<Path>>(
        &mut self,
        dst: P,
        dst_canonicalized: &Path,
        validated_paths: &mut FxHashSet<PathBuf>,
    ) -> io::Result<()> {
        self.fields
            .unpack_in_raw(dst, dst_canonicalized, validated_paths)
            .await
    }

    /// Sets whether to unpack extended file attributes (xattrs) on Unix systems.
    ///
    /// The default is `true`.
    pub fn set_unpack_xattrs(&mut self, unpack_xattrs: bool) {
        self.fields.unpack_xattrs = unpack_xattrs;
    }

    /// Sets whether to preserve file permissions when unpacking.
    ///
    /// The default is `true`.
    pub fn set_preserve_permissions(&mut self, preserve: bool) {
        self.fields.preserve_permissions = preserve;
    }

    /// Sets whether to preserve file ownerships when unpacking.
    ///
    /// The default is `true`.
    pub fn set_preserve_ownerships(&mut self, preserve: bool) {
        self.fields.preserve_ownerships = preserve;
    }

    /// Sets whether to preserve modification times when unpacking.
    ///
    /// The default is `true`.
    pub fn set_preserve_mtime(&mut self, preserve: bool) {
        self.fields.preserve_mtime = preserve;
    }

    /// Sets whether to overwrite existing files when unpacking.
    ///
    /// The default is `false`.
    pub fn set_overwrite(&mut self, overwrite: bool) {
        self.fields.overwrite = overwrite;
    }

    /// Sets whether to allow unpacking symlinks that point outside the target directory.
    ///
    /// # Security
    ///
    /// Setting this to `true` is a security risk when unpacking untrusted archives,
    /// as it could allow an attacker to create symlinks pointing to sensitive files
    /// outside the target directory.
    ///
    /// The default is `true` for backwards compatibility.
    pub fn set_allow_external_symlinks(&mut self, allow: bool) {
        self.fields.allow_external_symlinks = allow;
    }
}

impl<R: Read + Unpin> EntryFields<R> {
    pub fn from_header(
        pos: u64,
        header: Header,
        long_pathname: Option<Vec<u8>>,
        long_linkname: Option<Vec<u8>>,
        pax_extensions: Option<Vec<u8>>,
        size: u64,
        data: VecDeque<EntryIo<R>>,
    ) -> Self {
        Self {
            long_pathname,
            long_linkname,
            pax_extensions,
            header,
            size,
            header_pos: pos,
            file_pos: pos + 512,
            data,
            unpack_xattrs: true,
            preserve_permissions: true,
            preserve_ownerships: true,
            preserve_mtime: true,
            overwrite: false,
            allow_external_symlinks: true,
            read_state: None,
        }
    }

    async fn read_all(&mut self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(self.size as usize);
        self.read_to_end(&mut buf).await.map(|_| buf)
    }

    fn path(&self) -> io::Result<Cow<'_, Path>> {
        bytes2path(self.path_bytes()?)
    }

    fn path_bytes(&self) -> io::Result<Cow<'_, [u8]>> {
        match self.long_pathname {
            Some(ref bytes) => {
                if let Some(&0) = bytes.last() {
                    Ok(Cow::Borrowed(&bytes[..bytes.len() - 1]))
                } else {
                    Ok(Cow::Borrowed(bytes))
                }
            }
            None => {
                if let Some(ref pax) = self.pax_extensions {
                    // Check for malformed PAX extensions and return hard error
                    for ext in pax_extensions(pax) {
                        let ext = ext?; // Propagate error instead of silently dropping
                        if ext.key_bytes() == b"path" {
                            return Ok(Cow::Borrowed(ext.value_bytes()));
                        }
                    }
                }
                Ok(self.header.path_bytes())
            }
        }
    }

    /// Gets the path in a "lossy" way, used for error reporting ONLY.
    fn path_lossy(&self) -> String {
        // If path_bytes() fails, fall back to the header path for error reporting
        match self.path_bytes() {
            Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
            Err(_) => String::from_utf8_lossy(&self.header.path_bytes()).to_string(),
        }
    }

    fn link_name(&self) -> io::Result<Option<Cow<'_, Path>>> {
        match self.link_name_bytes()? {
            Some(bytes) => bytes2path(bytes).map(Some),
            None => Ok(None),
        }
    }

    fn link_name_bytes(&self) -> io::Result<Option<Cow<'_, [u8]>>> {
        match self.long_linkname {
            Some(ref bytes) => {
                if let Some(&0) = bytes.last() {
                    Ok(Some(Cow::Borrowed(&bytes[..bytes.len() - 1])))
                } else {
                    Ok(Some(Cow::Borrowed(bytes)))
                }
            }
            None => {
                if let Some(ref pax) = self.pax_extensions {
                    // Check for malformed PAX extensions and return hard error
                    for ext in pax_extensions(pax) {
                        let ext = ext?; // Propagate error instead of silently dropping
                        if ext.key_bytes() == b"linkpath" {
                            return Ok(Some(Cow::Borrowed(ext.value_bytes())));
                        }
                    }
                }
                Ok(self.header.link_name_bytes())
            }
        }
    }

    async fn pax_extensions(&mut self) -> io::Result<Option<PaxExtensions<'_>>> {
        if self.pax_extensions.is_none() {
            if !self.header.entry_type().is_pax_global_extensions()
                && !self.header.entry_type().is_pax_local_extensions()
            {
                return Ok(None);
            }
            self.pax_extensions = Some(self.read_all().await?);
        }
        Ok(Some(pax_extensions(self.pax_extensions.as_ref().unwrap())))
    }

    /// Unpack the [`Entry`] into the specified destination.
    ///
    /// It's assumed that `dst` is already canonicalized, and that the memoized set of validated
    /// paths are tied to `dst`.
    async fn unpack_in_raw<P: AsRef<Path>>(
        &mut self,
        dst: P,
        dst_canonicalized: &Path,
        validated_paths: &mut FxHashSet<PathBuf>,
    ) -> io::Result<()> {
        // These types are not supported yet, so return an error if they are encountered.
        match self.header.entry_type() {
            crate::EntryType::Block | crate::EntryType::Char | crate::EntryType::Fifo => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "special file types are not supported",
                ));
            }
            _ => {}
        }

        // Create the destination path
        let file_dst = dst.as_ref().join(
            self.path()
                .map_err(|e| {
                    TarError::new(
                        format!("invalid path in entry header: {}", self.path_lossy()),
                        e,
                    )
                })?
                .as_ref(),
        );

        // Normalize the path, removing any `.` or `..` components.
        let file_dst = normalize(file_dst).map_err(|e| {
            TarError::new(
                format!("invalid path in entry header: {}", self.path_lossy()),
                e,
            )
        })?;

        // Validate that the path doesn't escape the destination directory.
        // We check this by ensuring the normalized path is still within `dst_canonicalized`.
        if let Ok(normalized) = normalize_absolute(&file_dst) {
            // First, check the path is absolute and starts with the canonicalized destination.
            if !normalized.starts_with(dst_canonicalized) {
                return Err(TarError::new(
                    format!(
                        "entry path escapes the destination directory: {}",
                        self.path_lossy()
                    ),
                    io::Error::new(io::ErrorKind::InvalidData, "invalid path"),
                ));
            }

            // If the path hasn't been validated before, check it doesn't escape the
            // destination directory via symlinks.
            if !validated_paths.contains(&normalized) {
                // If `dst` is a symlink, validate that it points within the target directory.
                if let Some(parent) = file_dst.parent() {
                    if let Ok(metadata) = fs::symlink_metadata(parent).await {
                        if metadata.file_type().is_symlink() {
                            let canonicalized_parent = fs::canonicalize(parent).await.map_err(|e| {
                                TarError::new(
                                    format!("failed to canonicalize parent path: {}", parent.display()),
                                    e,
                                )
                            })?;

                            if !canonicalized_parent.starts_with(dst_canonicalized) {
                                return Err(TarError::new(
                                    format!(
                                        "symlink in entry path escapes the destination directory: {}",
                                        self.path_lossy()
                                    ),
                                    io::Error::new(io::ErrorKind::InvalidData, "invalid path"),
                                ));
                            }
                        }
                    }
                }

                validated_paths.insert(normalized);
            }
        }

        // Create parent directories if needed
        if let Some(parent) = file_dst.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                TarError::new(
                    format!("failed to create parent directory: {}", parent.display()),
                    e,
                )
            })?;
        }

        // Unpack based on entry type
        match self.header.entry_type() {
            crate::EntryType::Regular | crate::EntryType::Continuous => {
                self.unpack_regular(&file_dst).await?;
            }
            crate::EntryType::Symlink => {
                self.unpack_symlink(&file_dst, dst_canonicalized).await?;
            }
            crate::EntryType::HardLink => {
                self.unpack_hard_link(&file_dst, dst_canonicalized).await?;
            }
            crate::EntryType::Directory => {
                self.unpack_directory(&file_dst).await?;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!(
                        "unsupported entry type: {:?}",
                        self.header.entry_type()
                    ),
                ));
            }
        }

        Ok(())
    }

    /// Unpack the [`Entry`] into the specified destination.
    ///
    /// This function will determine what kind of file is pointed to by this
    /// entry and create it at the location `dst` with appropriate permissions
    /// if possible.
    pub async fn unpack_in<P: AsRef<Path>>(&mut self, dst: P) -> io::Result<()> {
        let dst = dst.as_ref();
        let dst_canonicalized = fs::canonicalize(dst).await.map_err(|e| {
            TarError::new(
                format!("failed to canonicalize destination: {}", dst.display()),
                e,
            )
        })?;
        let mut validated_paths = FxHashSet::default();
        self.unpack_in_raw(dst, &dst_canonicalized, &mut validated_paths)
            .await
    }

    /// Unpack the [`Entry`] into the specified destination.
    ///
    /// This function is provided as a convenience for easily extracting the
    /// contents of one entry into a particular destination.
    pub async fn unpack<P: AsRef<Path>>(&mut self, dst: P) -> io::Result<()> {
        self.unpack_in(dst).await
    }

    async fn unpack_regular(&self, file_dst: &Path) -> io::Result<()> {
        // If the file already exists and we're not overwriting, return an error.
        if !self.overwrite && fs::metadata(file_dst).await.is_ok() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("file already exists: {}", file_dst.display()),
            ));
        }

        // Create the file and write the contents.
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_dst)
            .await
            .map_err(|e| {
                TarError::new(
                    format!("failed to open file for writing: {}", file_dst.display()),
                    e,
                )
            })?;

        // Copy the entry's data to the file.
        let mut entry = self.clone();
        tokio::io::copy(&mut entry, &mut file).await.map_err(|e| {
            TarError::new(
                format!("failed to copy entry data to file: {}", file_dst.display()),
                e,
            )
        })?;

        // Set the file's permissions if requested.
        if self.preserve_permissions {
            let mode = self.header.mode()?;
            set_perms(file_dst, &mut file, mode, self.preserve_ownerships).await?;
        }

        // Set the file's mtime if requested.
        if self.preserve_mtime {
            let mtime = FileTime::from_unix_time(self.header.mtime()? as i64, 0);
            filetime::set_file_mtime(file_dst, mtime).map_err(|e| {
                TarError::new(
                    format!("failed to set file mtime: {}", file_dst.display()),
                    e,
                )
            })?;
        }

        Ok(())
    }

    async fn unpack_directory(&self, file_dst: &Path) -> io::Result<()> {
        // Create the directory if it doesn't exist.
        if fs::metadata(file_dst).await.is_err() {
            fs::create_dir(file_dst).await.map_err(|e| {
                TarError::new(
                    format!("failed to create directory: {}", file_dst.display()),
                    e,
                )
            })?;
        }

        // Set the directory's permissions if requested.
        if self.preserve_permissions {
            let mode = self.header.mode()?;
            let mut dummy = OpenOptions::new().read(true).open(file_dst).await.ok();
            set_perms(file_dst, dummy.as_mut(), mode, self.preserve_ownerships).await?;
        }

        // Set the directory's mtime if requested.
        if self.preserve_mtime {
            let mtime = FileTime::from_unix_time(self.header.mtime()? as i64, 0);
            filetime::set_file_mtime(file_dst, mtime).map_err(|e| {
                TarError::new(
                    format!("failed to set directory mtime: {}", file_dst.display()),
                    e,
                )
            })?;
        }

        Ok(())
    }

    async fn unpack_symlink(
        &mut self,
        file_dst: &Path,
        dst_canonicalized: &Path,
    ) -> io::Result<()> {
        // Get the link target.
        let link_target = self
            .link_name()
            .map_err(|e| {
                TarError::new(
                    format!("invalid link name in entry header: {}", self.path_lossy()),
                    e,
                )
            })?
            .ok_or_else(|| {
                TarError::new(
                    format!("missing link name in entry header: {}", self.path_lossy()),
                    io::Error::new(io::ErrorKind::InvalidData, "missing link name"),
                )
            })?;

        // If the file already exists and we're not overwriting, return an error.
        if !self.overwrite && fs::symlink_metadata(file_dst).await.is_ok() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("file already exists: {}", file_dst.display()),
            ));
        }

        // Remove the existing file if it exists.
        let _ = remove_file(file_dst).await;

        // Create the symlink.
        #[cfg(unix)]
        {
            let link_target = link_target.as_ref();
            tokio::fs::symlink(link_target, file_dst).await.map_err(|e| {
                TarError::new(
                    format!(
                        "failed to create symlink: {} -> {}",
                        file_dst.display(),
                        link_target.display()
                    ),
                    e,
                )
            })?;
        }
        #[cfg(windows)]
        {
            let link_target = link_target.as_ref();
            if self.is_dir() {
                tokio::fs::symlink_dir(link_target, file_dst).await.map_err(|e| {
                    TarError::new(
                        format!(
                            "failed to create directory symlink: {} -> {}",
                            file_dst.display(),
                            link_target.display()
                        ),
                        e,
                    )
                })?;
            } else {
                tokio::fs::symlink_file(link_target, file_dst).await.map_err(|e| {
                    TarError::new(
                        format!(
                            "failed to create file symlink: {} -> {}",
                            file_dst.display(),
                            link_target.display()
                        ),
                        e,
                    )
                })?;
            }
        }

        // Validate that the symlink target doesn't escape the destination directory.
        if !self.allow_external_symlinks {
            let link_target = link_target.as_ref();
            // Check if the link target is absolute or relative.
            if link_target.is_absolute() {
                // Absolute symlinks are always checked.
                let canonicalized_target = fs::canonicalize(link_target).await?;
                if !canonicalized_target.starts_with(dst_canonicalized) {
                    return Err(TarError::new(
                        format!(
                            "symlink target escapes the destination directory: {} -> {}",
                            self.path_lossy(),
                            link_target.display()
                        ),
                        io::Error::new(io::ErrorKind::InvalidData, "invalid symlink target"),
                    ));
                }
            } else {
                // For relative symlinks, resolve them relative to the symlink location.
                let resolved = file_dst.parent().unwrap_or(Path::new(".")).join(link_target);
                let normalized = normalize_relative(&resolved).map_err(|e| {
                    TarError::new(
                        format!(
                            "failed to normalize symlink target: {} -> {}",
                            self.path_lossy(),
                            link_target.display()
                        ),
                        e,
                    )
                })?;

                // If the normalized path is absolute, it must start with the destination.
                // If it's relative, we check if it escapes the parent directory.
                if normalized.is_absolute() {
                    if !normalized.starts_with(dst_canonicalized) {
                        return Err(TarError::new(
                            format!(
                                "symlink target escapes the destination directory: {} -> {}",
                                self.path_lossy(),
                                link_target.display()
                            ),
                            io::Error::new(io::ErrorKind::InvalidData, "invalid symlink target"),
                        ));
                    }
                } else {
                    // Relative path - check for parent directory escape.
                    for component in normalized.components() {
                        if matches!(component, Component::ParentDir) {
                            return Err(TarError::new(
                                format!(
                                    "symlink target escapes the destination directory: {} -> {}",
                                    self.path_lossy(),
                                    link_target.display()
                                ),
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "invalid symlink target",
                                ),
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn unpack_hard_link(
        &mut self,
        file_dst: &Path,
        dst_canonicalized: &Path,
    ) -> io::Result<()> {
        // Get the link target.
        let link_target = self
            .link_name()
            .map_err(|e| {
                TarError::new(
                    format!("invalid link name in entry header: {}", self.path_lossy()),
                    e,
                )
            })?
            .ok_or_else(|| {
                TarError::new(
                    format!("missing link name in entry header: {}", self.path_lossy()),
                    io::Error::new(io::ErrorKind::InvalidData, "missing link name"),
                )
            })?;

        // Resolve the link target relative to the destination directory.
        let link_target = dst_canonicalized.join(&link_target);

        // If the file already exists and we're not overwriting, return an error.
        if !self.overwrite && fs::metadata(file_dst).await.is_ok() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("file already exists: {}", file_dst.display()),
            ));
        }

        // Remove the existing file if it exists.
        let _ = remove_file(file_dst).await;

        // Create the hard link.
        tokio::fs::hard_link(&link_target, file_dst).await.map_err(|e| {
            TarError::new(
                format!(
                    "failed to create hard link: {} -> {}",
                    file_dst.display(),
                    link_target.display()
                ),
                e,
            )
        })?;

        Ok(())
    }
}

impl<R: Read + Unpin> Clone for EntryFields<R> {
    fn clone(&self) -> Self {
        Self {
            long_pathname: self.long_pathname.clone(),
            long_linkname: self.long_linkname.clone(),
            pax_extensions: self.pax_extensions.clone(),
            header: self.header.clone(),
            size: self.size,
            header_pos: self.header_pos,
            file_pos: self.file_pos,
            data: self.data.clone(),
            unpack_xattrs: self.unpack_xattrs,
            preserve_permissions: self.preserve_permissions,
            preserve_ownerships: self.preserve_ownerships,
            preserve_mtime: self.preserve_mtime,
            overwrite: self.overwrite,
            allow_external_symlinks: self.allow_external_symlinks,
            read_state: None,
        }
    }
}

impl<R: Read + Unpin> Clone for EntryIo<R> {
    fn clone(&self) -> Self {
        match self {
            EntryIo::Pad(t) => EntryIo::Pad(io::Repeat::new(0).take(t.limit())),
            EntryIo::Data(_) => panic!("cannot clone EntryIo::Data"),
        }
    }
}

impl<R: Read + Unpin> AsyncRead for Entry<R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut io::ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().fields).poll_read(cx, buf)
    }
}

impl<R: Read + Unpin> AsyncRead for EntryFields<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut io::ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        // Process any pending read state first
        if let Some(ref mut state) = self.read_state {
            let res = Pin::new(state).poll_read(cx, buf);
            if res.is_ready() {
                self.read_state = None;
            }
            return res;
        }

        // Get the next data chunk
        loop {
            match self.data.pop_front() {
                Some(EntryIo::Pad(mut pad)) => {
                    let res = Pin::new(&mut pad).poll_read(cx, buf);
                    if res.is_pending() {
                        self.read_state = Some(EntryIo::Pad(pad));
                        return Poll::Pending;
                    }
                }
                Some(EntryIo::Data(mut data)) => {
                    let res = Pin::new(&mut data).poll_read(cx, buf);
                    if res.is_pending() {
                        self.read_state = Some(EntryIo::Data(data));
                        return Poll::Pending;
                    }
                }
                None => return Poll::Ready(Ok(())),
            }
        }
    }
}

#[cfg(unix)]
async fn set_perms(
    dst: &Path,
    f: Option<&mut fs::File>,
    mode: u32,
    preserve_ownerships: bool,
) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    if preserve_ownerships {
        // Set the file's ownership.
        let uid = self.header.uid()?;
        let gid = self.header.gid()?;

        // Use the `libc` crate to set the file's ownership.
        let path_cstring = std::ffi::CString::new(dst.as_os_str().as_bytes())?;
        unsafe {
            if libc::chown(path_cstring.as_ptr(), uid as u32, gid as u32) != 0 {
                return Err(io::Error::last_os_error());
            }
        }
    }

    // Set the file's permissions.
    let perm = std::fs::Permissions::from_mode(mode);
    fs::set_permissions(dst, perm).await?;

    Ok(())
}

#[cfg(windows)]
async fn set_perms(
    dst: &Path,
    f: Option<&mut fs::File>,
    mode: u32,
    preserve_ownerships: bool,
) -> io::Result<()> {
    // On Windows, we only set the read-only flag based on the mode.
    if mode & 0o200 == 0 {
        // Read-only
        let mut perm = fs::metadata(dst).await?.permissions();
        perm.set_readonly(true);
        fs::set_permissions(dst, perm).await?;
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
async fn set_perms(
    dst: &Path,
    f: Option<&mut fs::File>,
    mode: u32,
    preserve_ownerships: bool,
) -> io::Result<()> {
    Ok(())
}
