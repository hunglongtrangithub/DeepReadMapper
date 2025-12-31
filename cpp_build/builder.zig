const std = @import("std");
const logger = @import("logger.zig");
const parser = @import("parser.zig");

const Log = logger.Log;
const DepIterator = parser.DepIterator;

/// Builder struct that encapsulates all build functionality
/// for compiling and linking C++ code with incremental builds.
pub const CppBuilder = struct {
    /// Zig build instance
    builder: *std.Build,
    /// Path to g++ compiler
    gxx_path: []const u8,
    /// Conda environment prefix path
    conda_prefix: []const u8,
    /// Command step to create binary output directory
    mkdir_bin_cmd: *std.Build.Step.Run,
    /// Command step to create object output directory
    mkdir_obj_cmd: *std.Build.Step.Run,
    /// Output directory for binaries
    bin_out: []const u8,
    /// Output directory for object files and dependency files
    obj_out: []const u8,

    const Self = @This();

    /// Initialize a new Builder instance
    ///
    /// Parameters:
    /// - `builder`: Zig's builder instance
    /// - `gxx_path`: Absolute path to the g++ executable
    /// - `conda_prefix`: Absolute path to the conda prefix (the currently active conda environment)
    /// - `bin_out`: The path to the directory that will contain binary files (either absolute or relative to `build.zig`'s directory)
    /// - `obj_out`: The path to the directory that will contain object files (either absolute or relative to `build.zig`'s directory)
    pub fn init(builder: *std.Build, gxx_path: []const u8, conda_prefix: []const u8, bin_out: []const u8, obj_out: []const u8) Self {
        const mkdir_bin_cmd = builder.addSystemCommand(&[_][]const u8{ "mkdir", "-p", bin_out });
        const mkdir_obj_cmd = builder.addSystemCommand(&[_][]const u8{ "mkdir", "-p", obj_out });

        return Self{
            .builder = builder,
            .gxx_path = gxx_path,
            .conda_prefix = conda_prefix,
            .mkdir_bin_cmd = mkdir_bin_cmd,
            .mkdir_obj_cmd = mkdir_obj_cmd,
            .bin_out = bin_out,
            .obj_out = obj_out,
        };
    }

    /// Convert source path to guaranteed-unique object file name (without extension) using SHA256 hash
    ///
    /// Format: {hash}_{basename}
    pub fn sourceToObjectName(self: Self, source_path: []const u8) []const u8 {
        // Create a hash of the full source path
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(source_path, &digest, .{});

        // Convert to hex string
        const short_slice = digest[0..16]; // Use first 16 bytes -> 32 hex chars
        const hex_hash = std.fmt.bytesToHex(short_slice, .lower);
        // Get the base filename (without extension) for readability
        const basename = std.fs.path.stem(std.fs.path.basename(source_path));

        // Combine: {hash}_{basename}
        return self.builder.fmt("{s}_{s}", .{ hex_hash, basename });
    }

    /// Check if object file needs rebuilding based on timestamps
    ///
    /// Returns true if:
    /// 1. Source file does not exist
    /// 2. Object file or dependency file does not exist
    /// 3. Source file is newer than object file
    /// 4. Any header dependency is newer than object file
    ///
    /// Otherwise returns false
    fn needsRebuild(self: Self, source_path: []const u8) !bool {
        // Get object file name based on source path
        const obj_name = self.sourceToObjectName(source_path);

        // Check if object file exists
        const obj_path = self.builder.fmt("{s}/{s}.o", .{ self.obj_out, obj_name });
        const obj_stat = std.fs.cwd().statFile(obj_path) catch {
            Log.debug("Object file missing: {s}", .{obj_path});
            return true;
        };

        // Check source file timestamp against object file
        const src_stat = std.fs.cwd().statFile(source_path) catch {
            Log.err("Source file not found: {s}", .{source_path});
            std.process.exit(1);
        };

        // If source is newer, need to rebuild
        if (src_stat.mtime > obj_stat.mtime) {
            Log.debug("Source {s} newer than object {s}", .{ source_path, obj_path });
            return true;
        }

        // Check dependency file for included headers
        const dep_path = self.builder.fmt("{s}/{s}.d", .{ self.obj_out, obj_name });
        _ = std.fs.cwd().statFile(dep_path) catch {
            Log.debug("Dependency file missing: {s}", .{dep_path});
            return true;
        };

        // Parse dependency file if it exists
        const needs_rebuild = try self.checkDependencyTimestamps(dep_path, obj_stat.mtime);
        if (needs_rebuild) {
            Log.debug("Dependencies changed for {s}", .{source_path});
        }
        return needs_rebuild;
    }

    /// Parse a dependency (.d) file and check if any dependency is newer than the object file.
    ///
    /// Parameters:
    /// - `dep_path`: Path to the dependency file.
    /// - `obj_mtime`: Modification time of the object file.
    ///
    /// Returns: true if any dependency is newer than the object file, false otherwise.
    fn checkDependencyTimestamps(self: Self, file_path: []const u8, obj_mtime: i128) !bool {
        const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
            Log.debug("Could not open dependency file: {s}", .{file_path});
            return err;
        };
        defer file.close();

        // Buffer to hold file content
        var file_content_buf = std.array_list.Managed(u8).init(self.builder.allocator);
        defer file_content_buf.deinit();

        // Read the entire file content
        var reader_buf: [1024]u8 = undefined;
        while (true) {
            const bytes_read = file.readAll(&reader_buf) catch |err| {
                Log.debug("Failed to read dependency file: {s}", .{file_path});
                return err;
            };

            // Reached end of file
            if (bytes_read == 0) break;

            // Append all read bytes to logical line buffer
            try file_content_buf.appendSlice(reader_buf[0..bytes_read]);
        }

        // Get the full file content as a slice
        const file_content = try file_content_buf.toOwnedSlice();

        Log.debug("Checking dependencies in {s}", .{file_path});
        var dep_iter = DepIterator.init(file_content, self.builder.allocator) catch |err| {
            Log.debug("Failed to parse dependency file: {s}", .{file_path});
            return err;
        };

        while (try dep_iter.next()) |dep| {
            // This is a dependency file - check its timestamp
            const dep_stat = std.fs.cwd().statFile(dep) catch {
                // If dependency file doesn't exist, force rebuild
                Log.debug("Dependency file missing: {s}", .{dep});
                return true;
            };

            if (dep_stat.mtime > obj_mtime) {
                // Dependency is newer than object file, need to rebuild
                Log.debug("Dependency {s} is newer than object", .{dep});
                return true;
            }
        }

        return false; // No dependencies are newer
    }

    /// Create a conditional object compilation step.
    ///
    /// Checks if the source file or its dependencies require recompilation.
    /// Returns null if no rebuild is needed, otherwise returns the build step for compilation.
    ///
    /// Parameters:
    /// - `source`: Path to the source file to compile.
    /// - `flags`: List of compiler flags to use.
    /// - `includes`: List of include directory flags.
    ///
    /// Returns: Nullable pointer to the build step for compiling the object file.
    pub fn createConditionalObjectCmd(self: Self, source: []const u8, flags: []const []const u8, includes: []const []const u8) !?*std.Build.Step.Run {
        // Only create compilation step if rebuild is needed
        if (!(try self.needsRebuild(source))) {
            Log.info("SKIP (up to date): {s}", .{source});
            return null;
        }

        Log.info("COMPILE: {s}", .{source});
        return self.createObjectCmd(source, flags, includes);
    }

    /// Create a system command to compile a source file into an object file.
    ///
    /// Parameters:
    /// - `source`: Path to the source file to compile.
    /// - `flags`: List of compiler flags to use.
    /// - `includes`: List of include directory flags.
    ///
    /// Returns: A pointer to the build step for compiling the object file.
    fn createObjectCmd(self: Self, source: []const u8, flags: []const []const u8, includes: []const []const u8) *std.Build.Step.Run {
        const cmd = self.builder.addSystemCommand(&[_][]const u8{self.gxx_path});

        cmd.addArgs(flags);

        // Compile only
        cmd.addArg("-c");

        cmd.addArgs(includes);

        cmd.addArg(self.builder.fmt("-isystem{s}/include", .{self.conda_prefix}));

        cmd.addArg(source);

        // Get object file name based on source path
        const obj_name = self.sourceToObjectName(source);
        cmd.addArgs(&[_][]const u8{ "-o", self.builder.fmt("{s}/{s}.o", .{ self.obj_out, obj_name }) });

        // Add dependency file generation for incremental builds
        cmd.addArg("-MMD"); // Generate dependency file
        cmd.addArg("-MP"); // Prevent errors for missing headers
        cmd.addArg(self.builder.fmt("-MF{s}/{s}.d", .{ self.obj_out, obj_name }));

        return cmd;
    }

    /// Create a system command to link object files into an executable.
    ///
    /// Parameters:
    /// - `flags`: List of linker flags to use.
    /// - `all_object_files`: List of object file paths to link.
    /// - `output_name`: Name of the output executable.
    /// - `extra_libs`: List of extra libraries to link against.
    ///
    /// Returns: A pointer to the build step for linking the executable.
    fn createLinkCmd(self: Self, flags: []const []const u8, all_object_files: []const []const u8, output_name: []const u8, extra_libs: []const []const u8) *std.Build.Step.Run {
        const cmd = self.builder.addSystemCommand(&[_][]const u8{self.gxx_path});

        cmd.addArgs(flags);

        cmd.addArgs(all_object_files);

        // tell linker where to find libraries at link-time
        cmd.addArg(self.builder.fmt("-L{s}/lib", .{self.conda_prefix}));
        // tell the executable where to find libraries at runtime
        cmd.addArg(self.builder.fmt("-Wl,-rpath,{s}/lib", .{self.conda_prefix}));

        cmd.addArgs(extra_libs);

        cmd.addArgs(&[_][]const u8{ "-o", self.builder.fmt("{s}/{s}", .{ self.bin_out, output_name }) });

        return cmd;
    }

    /// Build executable with incremental compilation
    ///
    /// Parameters:
    /// - `sources`: list of source files to compile
    /// - `common_objs`: list of precompiled object files to link against
    /// - `common_obj_steps`: list of build steps for common object files
    /// - `output`: name of the output executable
    /// - `libs`: list of libraries to link against
    /// - `flags`: list of flags for both compilation and linking
    /// - `includes`: list of include directories
    ///
    /// Returns: build step for linking the executable
    pub fn build(
        self: Self,
        sources: []const []const u8,
        common_objs: []const []const u8,
        common_obj_steps: []const *std.Build.Step.Run,
        output: []const u8,
        libs: []const []const u8,
        flags: []const []const u8,
        includes: []const []const u8,
    ) !*std.Build.Step.Run {
        var exe_objects = std.array_list.Managed([]const u8).init(self.builder.allocator);
        var exe_obj_steps = std.array_list.Managed(*std.Build.Step.Run).init(self.builder.allocator);

        // Add common objects
        for (common_objs) |obj| {
            try exe_objects.append(obj);
        }

        // Compile sources with incremental checking
        for (sources) |source| {
            const obj_path = self.builder.fmt("{s}/{s}.o", .{ self.obj_out, self.sourceToObjectName(source) });
            try exe_objects.append(obj_path);

            if (try self.createConditionalObjectCmd(source, flags, includes)) |obj_cmd| {
                obj_cmd.step.dependOn(&self.mkdir_obj_cmd.step);
                try exe_obj_steps.append(obj_cmd);
            }
        }

        const link_cmd = self.createLinkCmd(flags, exe_objects.items, output, libs);
        link_cmd.step.dependOn(&self.mkdir_bin_cmd.step);

        // Depend on common object compilations
        for (common_obj_steps) |obj_step| {
            link_cmd.step.dependOn(&obj_step.step);
        }

        // Depend on source-specific object compilations
        for (exe_obj_steps.items) |obj_step| {
            link_cmd.step.dependOn(&obj_step.step);
        }

        return link_cmd;
    }
};
