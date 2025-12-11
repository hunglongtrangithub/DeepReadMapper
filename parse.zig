const std = @import("std");

var stdout_buffer: [1024]u8 = undefined;
var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
const stdout = &stdout_writer.interface;

fn parse_dep_file(file_path: []const u8, allocator: std.mem.Allocator) ![][]const u8 {
    const file = std.fs.cwd().openFile(file_path, .{}) catch {
        std.debug.print("Could not open dependency file: {s}", .{file_path});
        return &.{};
    };
    defer file.close();

    // Buffer to hold file content
    var file_content_buf = std.array_list.Managed(u8).init(allocator);
    defer file_content_buf.deinit();

    // Read the entire file content
    var reader_buf: [1024]u8 = undefined;
    while (true) {
        const bytes_read = file.readAll(&reader_buf) catch {
            std.debug.print("Failed to read dependency file: {s}", .{file_path});
            return &.{};
        };

        // Reached end of file
        if (bytes_read == 0) break;

        // Append all read bytes to logical line buffer
        try file_content_buf.appendSlice(reader_buf[0..bytes_read]);
    }

    // Get the full file content as a slice
    const file_content = file_content_buf.items;
    std.debug.print("Dependency file content:\n{s}\n", .{file_content});

    return parse_dep_content(file_content, allocator);
}

fn parse_dep_content(file_content: []const u8, allocator: std.mem.Allocator) ![][]const u8 {
    // Find the start byte of the first dependency
    const dep_start = find_dep_start: {
        // Find the first colon that is followed by a space
        const first_colon_space = blk: {
            if (std.mem.indexOf(u8, file_content, ": ")) |pos| {
                break :blk pos + 2; // Start after ": "
            } else {
                return &.{}; // Not a valid dependency file
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
                // No dependencies found
                return &.{};
            }
        };
        break :find_dep_start actual_dep_start;
    };

    // Get the content after dep_start
    // dep_start is guaranteed to be within bounds from above checks
    const after_target_content = file_content[dep_start..];
    std.debug.print("After target content:\n{s}\n", .{after_target_content});

    // List to hold parsed dependencies
    var list = std.array_list.Managed([]const u8).init(allocator);

    // Parse each dependency, while handling backslash escaping and line continuation
    var dep = std.array_list.Managed(u8).init(allocator);
    defer dep.deinit();

    var last_is_backslash = false;
    for (after_target_content) |c| {
        switch (c) {
            '\\' => {
                if (last_is_backslash) {
                    // Consecutive backslashes are considered part of the dependency, so
                    // "////" or "multi_slash\\\\.cpp" are valid dependency names
                    try dep.append(c);
                } else {
                    last_is_backslash = true;
                }
            },
            ' ', '\t' => {
                if (last_is_backslash) {
                    // Handle escaped space/tab
                    try dep.append(c);
                    last_is_backslash = false;
                } else {
                    // End of dependency
                    // dep.items.len can be zero if there are multiple spaces/tabs between dependencies,
                    // so we have to guard against that
                    // Add the dependency to the list
                    if (dep.items.len > 0) {
                        std.debug.print("Parsed dependency: {s}\n", .{dep.items});
                        const dep_str = try allocator.dupe(u8, dep.items);
                        try list.append(dep_str);
                        dep.clearAndFree();
                    }
                }
            },
            '\n' => {
                if (last_is_backslash) {
                    // Line continuation, do not add anything
                    last_is_backslash = false;
                } else {
                    // End of the dependency logical line. Add the last dependency if any and stop parsing.
                    if (dep.items.len > 0) {
                        const dep_str = try allocator.dupe(u8, dep.items);
                        std.debug.print("Last parsed dependency: {s}\n", .{dep_str});
                        try list.append(dep_str);
                        dep.clearAndFree();
                    }
                    break;
                }
            },
            else => {
                if (last_is_backslash) {
                    // Previous backslash was not escaping a space/tab or newline, so it is part of the dependency
                    try dep.append('\\');
                    last_is_backslash = false;
                }
                try dep.append(c);
            },
        }
    }
    std.debug.print("Parsed dependencies:\n", .{});
    for (list.items) |d| {
        std.debug.print(" - {s}\n", .{d});
    }
    return list.toOwnedSlice();
}

pub fn main() !void {
    const file_path = "test.d";
    try stdout.print("Parsing dependency file: {s}\n", .{file_path});
    const allocator = std.heap.page_allocator;
    const deps = try parse_dep_file(file_path, allocator);
    defer allocator.free(deps);
    for (deps) |dep| {
        try stdout.print("Found dependency: <{s}>\n", .{dep});
    }
    try stdout.flush();
}
