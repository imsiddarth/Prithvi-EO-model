#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

typedef struct {
    int id, bt, remaining_bt, priority;
    pthread_cond_t cond;
} Process;

pthread_mutex_t cpu_lock = PTHREAD_MUTEX_INITIALIZER;
int turn = -1; 

void* process_work(void* arg) {
    Process *p = (Process*)arg;
    while (p->remaining_bt > 0) {
        pthread_mutex_lock(&cpu_lock);
        while (turn != p->id) pthread_cond_wait(&p->cond, &cpu_lock);
        printf("   [CPU] Process %d executing (Remaining: %d)\n", p->id, p->remaining_bt);
        usleep(100000); 
        p->remaining_bt--;
        turn = -1;
        pthread_mutex_unlock(&cpu_lock);
        usleep(1000); 
    }
    return NULL;
}

void run_scheduler(Process p[], int n, int order[], char* name, int is_rr, int quantum) {
    printf("\n--- Starting %s Scheduling ---\n", name);
    int completed = 0;
    while (completed < n) {
        completed = 0;
        for (int i = 0; i < n; i++) {
            int idx = order[i];
            if (p[idx].remaining_bt > 0) {
                int limit = is_rr ? quantum : p[idx].remaining_bt;
                for(int q=0; q < limit && p[idx].remaining_bt > 0; q++) {
                    pthread_mutex_lock(&cpu_lock);
                    turn = p[idx].id;
                    pthread_cond_signal(&p[idx].cond);
                    pthread_mutex_unlock(&cpu_lock);
                    while(turn != -1) usleep(1000);
                }
            } else completed++;
        }
        if (!is_rr) break; // Non-preemptive only needs one pass through the order
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) { printf("Usage: ./simulator <Num>\n"); return 1; }
    int n = atoi(argv[1]);
    pthread_t threads[n];
    Process proc[n];
    int order[n];
    srand(time(NULL));

    for (int i = 0; i < n; i++) {
        proc[i].id = i;
        proc[i].bt = (rand() % 5) + 2;
        proc[i].priority = rand() % 5;
        pthread_cond_init(&proc[i].cond, NULL);
    }

    char* names[] = {"FCFS", "SJF", "Priority", "Round Robin (Q=2)"};
    for (int algo = 0; algo < 4; algo++) {
        for(int i=0; i<n; i++) {
            order[i] = i;
            proc[i].remaining_bt = proc[i].bt;
        }
        // Sorting logic for SJF and Priority
        if (algo == 1) { // SJF
            for(int i=0; i<n-1; i++) for(int j=0; j<n-i-1; j++)
                if(proc[order[j]].bt > proc[order[j+1]].bt) { int t=order[j]; order[j]=order[j+1]; order[j+1]=t; }
        } else if (algo == 2) { // Priority
            for(int i=0; i<n-1; i++) for(int j=0; j<n-i-1; j++)
                if(proc[order[j]].priority > proc[order[j+1]].priority) { int t=order[j]; order[j]=order[j+1]; order[j+1]=t; }
        }

        for(int i=0; i<n; i++) pthread_create(&threads[i], NULL, process_work, &proc[i]);
        run_scheduler(proc, n, order, names[algo], (algo == 3), 2);
        for(int i=0; i<n; i++) pthread_join(threads[i], NULL);
    }
    printf("\nAll simulations completed on HP OMEN (WSL).\n");
    return 0;
}
