#include <unistd.h>
#include <libgen.h>
#include <string.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    (void)argc;
    char path[4096];
    strncpy(path, argv[0], sizeof(path) - 1);
    path[sizeof(path) - 1] = '\0';

    // argv[0] is .app/Contents/MacOS/DriftEngine
    // We need to cd to .app/../../.. (the build directory)
    char* dir = dirname(path);  // .app/Contents/MacOS
    dir = dirname(dir);         // .app/Contents
    dir = dirname(dir);         // .app
    dir = dirname(dir);         // build/

    chdir(dir);
    execl("./apps/launcher/launcher", "launcher", NULL);
    perror("exec failed");
    return 1;
}
