// #include "newline_inside\n.cpp" // g++ does not support newlines in filenames
// #include "slash\inside.cpp" // g++ does not support backslashes in filenames
// #include "slash_after.cpp\\" // g++ does not support backslashes in filenames
// #include "carriage\rhere.cpp" // g++ does not support carriage returns in filenames
#include " space_before.cpp"
#include "multi_slash\\.cpp"  // g++ does not support backslashes in filenames
#include "space_after.cpp "
#include "spaces    here.cpp"
#include "tab	here.cpp"
#include "tab\there.cpp"  // this is interpreted as a backslash followed by 't', not a tab
