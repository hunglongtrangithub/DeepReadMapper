const std = @import("std");

/// Expected conda environment name that should be activated
const CONDA_ENV = "DeepReadMapper";
/// Highest-level build output directory
const ZIG_OUT = "zig-out";
/// Output directory for binaries
const BIN_OUT = ZIG_OUT ++ "/bin";
/// Output directory for object files and dependency files
const OBJ_OUT = ZIG_OUT ++ "/obj";

/// Logging levels for build output
const LogLevel = enum(u8) {
    silent = 0, // No output
    err = 1, // Only errors
    warn = 2, // Errors and warnings
    info = 3, // Errors, warnings, and info
    debug = 4, // All output including debug
    trace = 5, // Most verbose output

    pub fn fromString(s: []const u8) ?LogLevel {
        if (std.mem.eql(u8, s, "silent")) return .silent;
        if (std.mem.eql(u8, s, "error")) return .err;
        if (std.mem.eql(u8, s, "warn")) return .warn;
        if (std.mem.eql(u8, s, "info")) return .info;
        if (std.mem.eql(u8, s, "debug")) return .debug;
        if (std.mem.eql(u8, s, "trace")) return .trace;
        return null;
    }

    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .silent => "silent",
            .err => "error",
            .warn => "warn",
            .info => "info",
            .debug => "debug",
            .trace => "trace",
        };
    }
};

/// Global logger instance
var global_log_level: LogLevel = .info;

/// Logger functions
const Log = struct {
    pub fn setLevel(level: LogLevel) void {
        global_log_level = level;
    }

    pub fn err(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.err)) {
            std.debug.print("[ERROR] " ++ fmt ++ "\n", args);
        }
    }

    pub fn warn(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.warn)) {
            std.debug.print("[WARN] " ++ fmt ++ "\n", args);
        }
    }

    pub fn info(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.info)) {
            std.debug.print("[INFO] " ++ fmt ++ "\n", args);
        }
    }

    pub fn debug(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.debug)) {
            std.debug.print("[DEBUG] " ++ fmt ++ "\n", args);
        }
    }

    pub fn trace(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.trace)) {
            std.debug.print("[TRACE] " ++ fmt ++ "\n", args);
        }
    }
};

pub const DepIterator = struct {
    /// The unescaped content of the dependency file after the target
    dep_content: []const u8,
    /// The current position in the content
    pos: usize,
    /// Buffer for the current dependency being parsed
    dep_buf: std.array_list.Managed(u8),
    /// Whether the last character was a backslash
    last_is_backslash: bool,
    /// Whether the iterator has reached the end
    done: bool,

    /// Find the start of dependencies after the target
    /// If no dependencies are found or the file content is invalid, returns null
    pub fn init(file_content: []const u8, allocator: std.mem.Allocator) !DepIterator {
        // Find the start byte of the first dependency
        const dep_start = find_dep_start: {
            // Find the first colon that is followed by a space
            const first_colon_space = blk: {
                if (std.mem.indexOf(u8, file_content, ": ")) |pos| {
                    break :blk pos + 2; // Start after ": "
                } else {
                    return error.InvalidDependencyFile; // Not a valid dependency file
                }
            };

            // Skip any additional tabs or spaces after the colon + space pair
            const actual_dep_start = blk: {
                var i: usize = first_colon_space;
                while (i < file_content.len) : (i += 1) {
                    if (file_content[i] != ' ' and file_content[i] != '\t') {
                        // Stop at the first non-space/tab character
                        break :blk i;
                    }
                } else {
                    // Either first_colon_space >= logical_line.len
                    // or there are only spaces/tabs after the colon + space pair
                    // => No dependencies found
                    // Set iterator as done immediately
                    return DepIterator{
                        .dep_content = "",
                        .pos = 0,
                        .dep_buf = std.array_list.Managed(u8).init(allocator),
                        .last_is_backslash = false,
                        .done = true,
                    };
                }
            };
            break :find_dep_start actual_dep_start;
        };

        // Get the content after dep_start
        // dep_start is guaranteed to be within bounds from above checks
        const after_target_content = file_content[dep_start..];
        return DepIterator{
            .dep_content = after_target_content,
            .pos = 0,
            .dep_buf = std.array_list.Managed(u8).init(allocator),
            .last_is_backslash = false,
            .done = false,
        };
    }

    pub fn deinit(self: *DepIterator) void {
        self.dep_buf.deinit();
    }

    /// Get the next dependency from the iterator.
    /// Returns null when there are no more dependencies.
    /// The returned slice is owned by the iterator.
    pub fn next(self: *DepIterator) !?[]const u8 {
        if (self.done) return null;

        self.dep_buf.clearAndFree();
        self.last_is_backslash = false;

        const see_unescaped_newline: bool = blk: while (self.pos < self.dep_content.len) : (self.pos += 1) {
            const c = self.dep_content[self.pos];

            switch (c) {
                '\\' => {
                    if (self.last_is_backslash) {
                        // Consecutive backslashes are considered part of the dependency, so
                        // "////" or "multi_slash\\\\.cpp" are valid dependency names
                        try self.dep_buf.append(c);
                    } else {
                        self.last_is_backslash = true;
                    }
                },
                ' ', '\t' => {
                    if (self.last_is_backslash) {
                        // Handle escaped space/tab
                        try self.dep_buf.append(c);
                        self.last_is_backslash = false;
                    } else {
                        // End of dependency
                        // dep.items.len can be zero if there are multiple spaces/tabs between dependencies,
                        // so we have to guard against that
                        // Add the dependency to the list
                        if (self.dep_buf.items.len > 0) {
                            const dep_str = self.dep_buf.items;
                            Log.debug("Parsed dependency: {s}", .{dep_str});
                            return dep_str;
                        }
                    }
                },
                '\n' => {
                    if (self.last_is_backslash) {
                        // Line continuation, do not add anything
                        self.last_is_backslash = false;
                    } else {
                        // End of the dependency logical line. Add the last dependency if any and stop parsing.
                        if (self.dep_buf.items.len > 0) {
                            const dep_str = self.dep_buf.items;
                            Log.debug("Last parsed dependency: {s}", .{dep_str});
                            return dep_str;
                        }
                        break :blk true;
                    }
                },
                else => {
                    if (self.last_is_backslash) {
                        // Previous backslash was not escaping a space/tab or newline, so it is part of the dependency
                        try self.dep_buf.append('\\');
                        self.last_is_backslash = false;
                    }
                    try self.dep_buf.append(c);
                },
            }
        } else {
            break :blk false;
        };

        // End of content reached. Mark as done.
        self.done = true;
        if (self.dep_buf.items.len > 0) {
            const dep_str = self.dep_buf.items;
            if (see_unescaped_newline) {
                Log.debug("Last parsed dependency at end of logical line: {s}", .{dep_str});
            } else {
                Log.debug("Last parsed dependency at end of file: {s}", .{dep_str});
            }
            return dep_str;
        } else {
            return null;
        }
    }
};

/// Builder struct that encapsulates all build functionality
const Builder = struct {
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

    const Self = @This();

    /// Initialize a new Builder instance
    pub fn init(builder: *std.Build, gxx_path: []const u8, conda_prefix: []const u8) Self {
        const mkdir_bin_cmd = builder.addSystemCommand(&[_][]const u8{ "mkdir", "-p", BIN_OUT });
        const mkdir_obj_cmd = builder.addSystemCommand(&[_][]const u8{ "mkdir", "-p", OBJ_OUT });

        return Self{
            .builder = builder,
            .gxx_path = gxx_path,
            .conda_prefix = conda_prefix,
            .mkdir_bin_cmd = mkdir_bin_cmd,
            .mkdir_obj_cmd = mkdir_obj_cmd,
        };
    }

    /// Convert source path to guaranteed-unique object file name (without extension) using SHA256 hash
    ///
    /// Format: {basename}_{hash}
    fn sourceToObjectName(self: Self, source_path: []const u8) []const u8 {
        // Create a hash of the full source path
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(source_path, &digest, .{});

        // Convert to hex string
        const short_slice = digest[0..16]; // Use first 16 bytes -> 32 hex chars
        const hex_hash = std.fmt.bytesToHex(short_slice, .lower);
        // Get the base filename (without extension) for readability
        const basename = std.fs.path.stem(std.fs.path.basename(source_path));

        // Combine: {basename}_{hash}
        return self.builder.fmt("{s}_{s}", .{ basename, hex_hash });
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
        const obj_path = self.builder.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name });
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
        const dep_path = self.builder.fmt("{s}/{s}.d", .{ OBJ_OUT, obj_name });
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
    fn createConditionalObjectCmd(self: Self, source: []const u8, flags: []const []const u8, includes: []const []const u8) !?*std.Build.Step.Run {
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

        for (flags) |flag| {
            cmd.addArg(flag);
        }

        // Compile only
        cmd.addArg("-c");

        for (includes) |inc| {
            cmd.addArg(inc);
        }

        cmd.addArg(self.builder.fmt("-isystem{s}/include", .{self.conda_prefix}));
        cmd.addArg(source);

        // Get object file name based on source path
        const obj_name = self.sourceToObjectName(source);
        cmd.addArgs(&[_][]const u8{ "-o", self.builder.fmt("{s}/{s}.o", .{ OBJ_OUT, obj_name }) });

        // Add dependency file generation for incremental builds
        cmd.addArg("-MMD"); // Generate dependency file
        cmd.addArg("-MP"); // Prevent errors for missing headers
        cmd.addArg(self.builder.fmt("-MF{s}/{s}.d", .{ OBJ_OUT, obj_name }));

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

        for (all_object_files) |obj| {
            cmd.addArg(obj);
        }

        cmd.addArg(self.builder.fmt("-L{s}/lib", .{self.conda_prefix}));
        cmd.addArgs(&[_][]const u8{});

        for (extra_libs) |lib| {
            cmd.addArg(lib);
        }

        cmd.addArgs(&[_][]const u8{ "-o", self.builder.fmt("{s}/{s}", .{ BIN_OUT, output_name }) });

        return cmd;
    }

    /// Build executable with incremental compilation
    ///
    /// Parameters:
    /// - `sources`: list of source files to compile
    /// - `common_objs`: optional list of precompiled object files to link against
    /// - `common_obj_steps`: optional list of build steps for common object files
    /// - `output`: name of the output executable
    /// - `libs`: list of libraries to link against
    /// - `flags`: list of flags for both compilation and linking
    /// - `includes`: list of include directories
    ///
    /// Returns: build step for linking the executable
    pub fn build(
        self: Self,
        sources: []const []const u8,
        common_objs: ?[]const []const u8,
        common_obj_steps: ?[]const *std.Build.Step.Run,
        output: []const u8,
        libs: []const []const u8,
        flags: []const []const u8,
        includes: []const []const u8,
    ) !*std.Build.Step.Run {
        var exe_objects = std.array_list.Managed([]const u8).init(self.builder.allocator);
        var exe_obj_steps = std.array_list.Managed(*std.Build.Step.Run).init(self.builder.allocator);

        // Add common objects if provided
        if (common_objs) |objs| {
            for (objs) |obj| {
                try exe_objects.append(obj);
            }
        }

        // Compile sources with incremental checking
        for (sources) |source| {
            const obj_path = self.builder.fmt("{s}/{s}.o", .{ OBJ_OUT, self.sourceToObjectName(source) });
            try exe_objects.append(obj_path);

            if (try self.createConditionalObjectCmd(source, flags, includes)) |obj_cmd| {
                obj_cmd.step.dependOn(&self.mkdir_obj_cmd.step);
                try exe_obj_steps.append(obj_cmd);
            }
        }

        const link_cmd = self.createLinkCmd(flags, exe_objects.items, output, libs);
        link_cmd.step.dependOn(&self.mkdir_bin_cmd.step);

        // Depend on common object compilations if provided
        if (common_obj_steps) |steps| {
            for (steps) |obj_step| {
                link_cmd.step.dependOn(&obj_step.step);
            }
        }

        // Depend on source-specific object compilations
        for (exe_obj_steps.items) |obj_step| {
            link_cmd.step.dependOn(&obj_step.step);
        }

        return link_cmd;
    }
};

pub fn build(b: *std.Build) !void {
    // Configure logging level from command line
    const log_level_str = b.option([]const u8, "log", "Set log level (silent, error, warn, info, debug, trace)") orelse "info";
    const log_level = LogLevel.fromString(log_level_str) orelse {
        std.debug.print("Invalid log level: {s}. Valid levels: silent, error, warn, info, debug, trace\n", .{log_level_str});
        std.process.exit(1);
    };
    Log.setLevel(log_level);

    const conda_prefix = std.posix.getenv("CONDA_PREFIX") orelse {
        Log.err("CONDA_PREFIX not set. Please activate conda environment: {s}", .{CONDA_ENV});
        std.process.exit(1);
    };

    if (!std.mem.eql(u8, std.fs.path.basename(conda_prefix), CONDA_ENV)) {
        Log.err("Incorrect conda environment: {s}. Please activate conda environment: {s}", .{ conda_prefix, CONDA_ENV });
        std.process.exit(1);
    }

    Log.info("Using conda prefix: {s}", .{conda_prefix});

    const gxx_path = b.findProgram(&[_][]const u8{"g++"}, &[_][]const u8{
        b.fmt("{s}/bin", .{conda_prefix}),
        "/usr/bin",
    }) catch "g++";

    Log.info("Using g++ path: {s}", .{gxx_path});

    // Initialize Builder
    const builder = Builder.init(b, gxx_path, conda_prefix);

    const common_sources = [_][]const u8{
        "external/cnpy/cnpy.cpp",
        "src/utils/parse_inputs.cpp",
        "src/utils/utils.cpp",
        "src/utils/tok2index.cpp",
        "src/utils/post_processor.cpp",
        "src/utils/metrics.cpp",
        "src/utils/reranker.cpp",
        "src/inference/vectorize.cpp",
        "src/inference/fast_model.cpp",
        "src/inference/preprocess.cpp",
    };

    const common_flags = [_][]const u8{
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-march=native",
        "-Wall",
        "-lstdc++", // for C++ standard library
        "-lz", // zlib for cnpy
    };

    const common_includes = [_][]const u8{
        "-Iincludes",
        "-Iincludes/inference",
        "-Iincludes/utils",
        "-Iincludes/hnswlib_dir",
        "-Iincludes/gann_hnsw",
        "-Iincludes/hnswpq",
        "-Iexternal/cereal/include",
        "-Iexternal/cnpy",
    };

    // Compile common sources first to be used by multiple executables
    var all_object_files = std.array_list.Managed([]const u8).init(b.allocator);
    var all_object_steps = std.array_list.Managed(*std.Build.Step.Run).init(b.allocator);

    for (common_sources) |source| {
        const obj_path = b.fmt("{s}/{s}.o", .{ OBJ_OUT, builder.sourceToObjectName(source) });
        try all_object_files.append(obj_path);

        if (try builder.createConditionalObjectCmd(source, &common_flags, &common_includes)) |obj_cmd| {
            obj_cmd.step.dependOn(&builder.mkdir_obj_cmd.step);
            try all_object_steps.append(obj_cmd);
        }
    }

    // Build executables using the Builder
    const pipeline_cmd = try builder.build(
        &[_][]const u8{ "src/main.cpp", "src/hnswpq/search.cpp" },
        all_object_files.items,
        all_object_steps.items,
        "pipeline",
        &[_][]const u8{ "-lopenvino", "-lomp", "-lfaiss" },
        &common_flags,
        &common_includes,
    );

    const inference_cmd = try builder.build(
        &[_][]const u8{"src/inference/test_inference.cpp"},
        all_object_files.items,
        all_object_steps.items,
        "inference",
        &[_][]const u8{"-lopenvino"},
        &common_flags,
        &common_includes,
    );

    const hnswpq_cmd = try builder.build(
        &[_][]const u8{"src/hnswpq/index.cpp"},
        all_object_files.items,
        all_object_steps.items,
        "hnswpq_index",
        &[_][]const u8{ "-lopenvino", "-lomp", "-lfaiss" },
        &common_flags,
        &common_includes,
    );

    // Set up build step dependencies
    const install_step = b.getInstallStep();
    install_step.dependOn(&pipeline_cmd.step);
    install_step.dependOn(&inference_cmd.step);
    install_step.dependOn(&hnswpq_cmd.step);

    // Individual build steps
    const pipeline_step = b.step("pipeline", "Build only the pipeline executable");
    pipeline_step.dependOn(&pipeline_cmd.step);

    const inference_step = b.step("inference", "Build only the inference executable");
    inference_step.dependOn(&inference_cmd.step);

    const index_step = b.step("index", "Build only the index executable");
    index_step.dependOn(&hnswpq_cmd.step);

    // Build step for common object files
    const objects_step = b.step("objects", "Build common object files");
    for (all_object_steps.items) |obj_step| {
        objects_step.dependOn(&obj_step.step);
    }

    // Clean step to remove build artifacts
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", BIN_OUT, OBJ_OUT });
    clean_step.dependOn(&clean_cmd.step);
}

test "basic dependencies" {
    const allocator = std.testing.allocator;
    const content = "target.o: dep1.cpp dep2.h dep3.hpp\n";
    var it = try DepIterator.init(content, allocator);
    defer it.deinit();

    try std.testing.expectEqualStrings("dep1.cpp", (try it.next()).?);
    try std.testing.expectEqualStrings("dep2.h", (try it.next()).?);
    try std.testing.expectEqualStrings("dep3.hpp", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}

test "dependencies with spaces and tabs" {
    const allocator = std.testing.allocator;
    const content =
        "zig-out/obj/test:_b901a52b31646cdd4a0c2f7db1140fa3.o: \\\n" ++
        " tests/test_gxx_filenames/test:.cpp \\\n" ++
        " tests/test_gxx_filenames/\\ space_before.cpp \\\n" ++
        " tests/test_gxx_filenames/space_after.cpp\\  \\\n" ++
        " tests/test_gxx_filenames/spaces\\ \\ \\ \\ here.cpp \\\n" ++
        " tests/test_gxx_filenames/tab\\\there.cpp \\\n" ++
        " tests/test_gxx_filenames/tab\\there.cpp\n" ++
        "tests/test_gxx_filenames/\\ space_before.cpp:\n" ++
        "tests/test_gxx_filenames/space_after.cpp\\ :\n" ++
        "tests/test_gxx_filenames/spaces\\ \\ \\ \\ here.cpp:\n" ++
        "tests/test_gxx_filenames/tab\\\there.cpp:\n" ++
        "tests/test_gxx_filenames/tab\\there.cpp:\n";
    var it = try DepIterator.init(content, allocator);
    defer it.deinit();

    try std.testing.expectEqualStrings("tests/test_gxx_filenames/test:.cpp", (try it.next()).?);
    try std.testing.expectEqualStrings("tests/test_gxx_filenames/ space_before.cpp", (try it.next()).?);
    try std.testing.expectEqualStrings("tests/test_gxx_filenames/space_after.cpp ", (try it.next()).?);
    try std.testing.expectEqualStrings("tests/test_gxx_filenames/spaces    here.cpp", (try it.next()).?);
    try std.testing.expectEqualStrings("tests/test_gxx_filenames/tab\there.cpp", (try it.next()).?);
    try std.testing.expectEqualStrings("tests/test_gxx_filenames/tab\\there.cpp", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}

test "escaped spaces" {
    const allocator = std.testing.allocator;
    const content = "target.o: dep\\ with\\ space.cpp another.h\n";
    var it = try DepIterator.init(content, allocator);
    defer it.deinit();

    try std.testing.expectEqualStrings("dep with space.cpp", (try it.next()).?);
    try std.testing.expectEqualStrings("another.h", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}

test "line continuation" {
    const allocator = std.testing.allocator;
    const content = "target.o: dep1.cpp \\\n dep2.h \\\n dep3.hpp \\\n";
    var it = try DepIterator.init(content, allocator);
    defer it.deinit();

    try std.testing.expectEqualStrings("dep1.cpp", (try it.next()).?);
    try std.testing.expectEqualStrings("dep2.h", (try it.next()).?);
    try std.testing.expectEqualStrings("dep3.hpp", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}

test "escaped tabs" {
    const allocator = std.testing.allocator;
    const content = "target.o: file\\\twith\\\ttab.cpp\n";
    var it = try DepIterator.init(content, allocator);
    defer it.deinit();

    try std.testing.expectEqualStrings("file\twith\ttab.cpp", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}

test "consecutive backslashes" {
    const allocator = std.testing.allocator;
    const content = "target.o: path\\\\with\\\\backslash.cpp\n";
    var it = try DepIterator.init(content, allocator);
    defer it.deinit();

    try std.testing.expectEqualStrings("path\\\\with\\\\backslash.cpp", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}

test "multiple spaces and tabs" {
    const allocator = std.testing.allocator;
    const content = "target.o: dep1.cpp        dep2.h      dep3.hpp\n";
    var it = try DepIterator.init(content, allocator);
    defer it.deinit();

    try std.testing.expectEqualStrings("dep1.cpp", (try it.next()).?);
    try std.testing.expectEqualStrings("dep2.h", (try it.next()).?);
    try std.testing.expectEqualStrings("dep3.hpp", (try it.next()).?);
    try std.testing.expect((try it.next()) == null);
}

test "empty dependency list" {
    const allocator = std.testing.allocator;
    const content = "target.o: \n";
    var it = try DepIterator.init(content, allocator);
    defer it.deinit();

    try std.testing.expect((try it.next()) == null);
}

test "no colon followed by space" {
    const allocator = std.testing.allocator;
    const invalid_contents: [4][]const u8 = .{ "invalid content without colon\n", "target.o:\n", "target.o\n", "target.o dep1.cpp\n" };
    for (invalid_contents) |invalid_content| {
        try std.testing.expectError(error.InvalidDependencyFile, DepIterator.init(invalid_content, allocator));
    }
}

test "no end of logical line" {
    const allocator = std.testing.allocator;
    const content = "target.o: dep1.cpp dep2.h";
    var it = try DepIterator.init(content, allocator);
    defer it.deinit();

    try std.testing.expectEqualStrings("dep1.cpp", (try it.next()).?);
    try std.testing.expectEqualStrings("dep2.h", (try it.next()).?);
}
