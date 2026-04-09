#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    pid_t pid = fork();

    if (pid < 0) {
        return 1;
    } else if (pid == 0) {
        // Child logic
        printf("Child (PID %d) Sequence: ", getpid());
        while (n >= 1) {
            printf("%d ", n);
            n /= 2;
        }
        printf("\n");
    } else {
        // Parent logic
        wait(NULL);
        printf("Parent (PID %d): Child is finished.\n", getpid());
    }
    return 0;
}
