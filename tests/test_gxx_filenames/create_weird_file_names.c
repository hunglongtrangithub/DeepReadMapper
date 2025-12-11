#include <stdio.h>

int main() {
    const char* filenames[] = {" space_before.cpp",  "multi_slash\\\\.cpp\n", "space_after.cpp ",
                               "spaces    here.cpp", "tab\there.cpp",         "tab\\there.cpp",
                               "carriage\rhere.cpp", "newline_inside\n.cpp"};
    for (int i = 0; i < sizeof(filenames) / sizeof(filenames[0]); i++) {
        const char* filename = filenames[i];
        FILE* file = fopen(filename, "a+");
        if (file) {
            printf("Successfully opened file: %s\n", filename);
            fclose(file);
        } else {
            printf("Failed to open file: %s\n", filename);
        }
    }
}
