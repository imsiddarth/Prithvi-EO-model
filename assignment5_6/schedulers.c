#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

typedef struct {
    int id;
    int arrival_time;
    int burst_time;
    int remaining_time;
    int priority;
    int completion_time;
    int turnaround_time;
    int waiting_time;
    int response_time;
    int started;
    int first_start_time;
} Process;

// Helper to reset process data between algorithm runs
void reset_processes(Process p[], int n) {
    for (int i = 0; i < n; i++) {
        p[i].remaining_time = p[i].burst_time;
        p[i].started = 0;
        p[i].completion_time = 0;
        p[i].turnaround_time = 0;
        p[i].waiting_time = 0;
        p[i].response_time = 0;
    }
}

// 1. FCFS
void run_fcfs(Process p[], int n, FILE *f) {
    fprintf(f, "\n================ FCFS SCHEDULING ================\n");
    int current_time = 0;
    float sum_wt = 0, sum_tat = 0, sum_rt = 0;

    for (int i = 0; i < n; i++) {
        if (current_time < p[i].arrival_time) current_time = p[i].arrival_time;
        p[i].response_time = current_time - p[i].arrival_time;
        current_time += p[i].burst_time;
        p[i].completion_time = current_time;
        p[i].turnaround_time = p[i].completion_time - p[i].arrival_time;
        p[i].waiting_time = p[i].turnaround_time - p[i].burst_time;

        sum_wt += p[i].waiting_time; sum_tat += p[i].turnaround_time; sum_rt += p[i].response_time;
        fprintf(f, "P%d\t| AT:%d\tBT:%d\tTAT:%d\tWT:%d\tRT:%d\n", p[i].id, p[i].arrival_time, p[i].burst_time, p[i].turnaround_time, p[i].waiting_time, p[i].response_time);
    }
    fprintf(f, "AVERAGE: WT:%.2f  TAT:%.2f  RT:%.2f\n", sum_wt/n, sum_tat/n, sum_rt/n);
}

// 2. Round Robin
void run_rr(Process p[], int n, int q, FILE *f) {
    fprintf(f, "\n================ ROUND ROBIN (Q:%d) ================\n", q);
    reset_processes(p, n);
    int current_time = 0, completed = 0;
    float sum_wt = 0, sum_tat = 0, sum_rt = 0;

    while (completed < n) {
        int flag = 0;
        for (int i = 0; i < n; i++) {
            if (p[i].arrival_time <= current_time && p[i].remaining_time > 0) {
                flag = 1;
                if (!p[i].started) {
                    p[i].response_time = current_time - p[i].arrival_time;
                    p[i].started = 1;
                }
                if (p[i].remaining_time > q) {
                    current_time += q;
                    p[i].remaining_time -= q;
                } else {
                    current_time += p[i].remaining_time;
                    p[i].remaining_time = 0;
                    p[i].completion_time = current_time;
                    p[i].turnaround_time = p[i].completion_time - p[i].arrival_time;
                    p[i].waiting_time = p[i].turnaround_time - p[i].burst_time;
                    sum_wt += p[i].waiting_time; sum_tat += p[i].turnaround_time; sum_rt += p[i].response_time;
                    completed++;
                    fprintf(f, "P%d\t| TAT:%d\tWT:%d\tRT:%d\n", p[i].id, p[i].turnaround_time, p[i].waiting_time, p[i].response_time);
                }
            }
        }
        if (!flag) current_time++;
    }
    fprintf(f, "AVERAGE: WT:%.2f  TAT:%.2f  RT:%.2f\n", sum_wt/n, sum_tat/n, sum_rt/n);
}

// 3. SJF (All arrive at 0)
void run_sjf(Process p[], int n, FILE *f) {
    fprintf(f, "\n================ SJF (Arrival at 0) ================\n");
    reset_processes(p, n);
    // Sort by Burst Time
    for(int i=0; i<n-1; i++)
        for(int j=0; j<n-i-1; j++)
            if(p[j].burst_time > p[j+1].burst_time) {
                Process temp = p[j]; p[j] = p[j+1]; p[j+1] = temp;
            }
    
    int current_time = 0;
    float sum_wt = 0, sum_tat = 0;
    for (int i = 0; i < n; i++) {
        p[i].waiting_time = current_time;
        p[i].response_time = current_time;
        current_time += p[i].burst_time;
        p[i].turnaround_time = current_time;
        sum_wt += p[i].waiting_time; sum_tat += p[i].turnaround_time;
        fprintf(f, "P%d\t| BT:%d\tTAT:%d\tWT:%d\n", p[i].id, p[i].burst_time, p[i].turnaround_time, p[i].waiting_time);
    }
    fprintf(f, "AVERAGE: WT:%.2f  TAT:%.2f\n", sum_wt/n, sum_tat/n);
}

// 4. SRTN (Shortest Remaining Time Next)
void run_srtn(Process p[], int n, FILE *f) {
    fprintf(f, "\n================ SRTN SCHEDULING ================\n");
    reset_processes(p, n);
    int current_time = 0, completed = 0;
    float sum_wt = 0, sum_tat = 0, sum_rt = 0;

    while (completed < n) {
        int idx = -1, min_bt = 1e9;
        for (int i = 0; i < n; i++) {
            if (p[i].arrival_time <= current_time && p[i].remaining_time > 0 && p[i].remaining_time < min_bt) {
                min_bt = p[i].remaining_time;
                idx = i;
            }
        }
        if (idx != -1) {
            if (!p[idx].started) {
                p[idx].response_time = current_time - p[idx].arrival_time;
                p[idx].started = 1;
            }
            p[idx].remaining_time--;
            current_time++;
            if (p[idx].remaining_time == 0) {
                p[idx].completion_time = current_time;
                p[idx].turnaround_time = p[idx].completion_time - p[idx].arrival_time;
                p[idx].waiting_time = p[idx].turnaround_time - p[idx].burst_time;
                sum_wt += p[idx].waiting_time; sum_tat += p[idx].turnaround_time; sum_rt += p[idx].response_time;
                completed++;
                fprintf(f, "P%d\t| TAT:%d\tWT:%d\tRT:%d\n", p[idx].id, p[idx].turnaround_time, p[idx].waiting_time, p[idx].response_time);
            }
        } else current_time++;
    }
    fprintf(f, "AVERAGE: WT:%.2f  TAT:%.2f  RT:%.2f\n", sum_wt/n, sum_tat/n, sum_rt/n);
}

// 5. Priority (Non-Preemptive)
void run_priority(Process p[], int n, FILE *f) {
    fprintf(f, "\n================ PRIORITY SCHEDULING ================\n");
    reset_processes(p, n);
    int current_time = 0, completed = 0;
    float sum_wt = 0, sum_tat = 0;

    while (completed < n) {
        int idx = -1, highest_prio = 1e9;
        for (int i = 0; i < n; i++) {
            if (p[i].arrival_time <= current_time && !p[i].started && p[i].priority < highest_prio) {
                highest_prio = p[i].priority;
                idx = i;
            }
        }
        if (idx != -1) {
            p[idx].started = 1;
            p[idx].waiting_time = current_time - p[idx].arrival_time;
            current_time += p[idx].burst_time;
            p[idx].turnaround_time = current_time - p[idx].arrival_time;
            sum_wt += p[idx].waiting_time; sum_tat += p[idx].turnaround_time;
            completed++;
            fprintf(f, "P%d\t| Prio:%d\tTAT:%d\tWT:%d\n", p[idx].id, p[idx].priority, p[idx].turnaround_time, p[idx].waiting_time);
        } else current_time++;
    }
    fprintf(f, "AVERAGE: WT:%.2f  TAT:%.2f\n", sum_wt/n, sum_tat/n);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: ./CPUSimulator <Algo_Choice> <Process_Count>\n");
        return 1;
    }

    int choice = atoi(argv[1]);
    int n = atoi(argv[2]);
    Process proc[n];
    srand(time(NULL));

    FILE *fData = fopen("process_data.txt", "w");
    FILE *fOut = fopen("Output.txt", "w");

    fprintf(fData, "ID\tAT\tBT\tPrio\n");
    for (int i = 0; i < n; i++) {
        proc[i].id = i + 1;
        proc[i].arrival_time = rand() % 20;
        proc[i].burst_time = (rand() % 15) + 1;
        proc[i].priority = rand() % 10;
        fprintf(fData, "%d\t%d\t%d\t%d\n", proc[i].id, proc[i].arrival_time, proc[i].burst_time, proc[i].priority);
    }
    fclose(fData);

    printf("Hi! Welcome to CPU Scheduling Simulator.\nWait.... Generating Schedules...\n");

    if (choice == 1 || choice == 6) run_fcfs(proc, n, fOut);
    if (choice == 2 || choice == 6) run_rr(proc, n, 4, fOut);
    if (choice == 3 || choice == 6) run_sjf(proc, n, fOut);
    if (choice == 4 || choice == 6) run_srtn(proc, n, fOut);
    if (choice == 5 || choice == 6) run_priority(proc, n, fOut);

    printf("DONE. Check Output.txt\n");
    fclose(fOut);
    return 0;
}
