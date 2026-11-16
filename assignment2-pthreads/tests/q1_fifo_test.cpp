// CS433 Assignment 2, Question 1: FIFO + fairness tests (not part of the solution)
// Group 1: Ashik Stenny, Spandan Pati, Vedhanth Balasubramanian
// reuses the semaphores from q1_mpsc.cpp as-is (its main renamed)
// build: g++ -O2 -DVERSION=k -o fifo_pk q1_fifo_test.cpp -lpthread
// usage: ./fifo_pk fifo T K        -> T threads, K wait/post each, semaphore = 1
//        ./fifo_pk fair N P n      -> MPSC run, per-producer finish times
#define main mpsc_main
#include "../q1_mpsc.cpp"
#undef main

#include <vector>

#define CS_WORK 50		// pause loops inside critical section -> queue builds up

my_sem_t sem;
int T, K;
int arrival = 0;		// arrival number, taken just before wait()
int *log_arrival;		// arrival numbers in pass order
int log_len = 0;		// only touched inside critical section

void *fifo_worker (void *param)
{
	(void)param;
	for (int i=0; i<K; i++) {
		int a = FetchAndInc(&arrival);
		my_sem_wait(&sem);
		log_arrival[log_len] = a;
		log_len++;
		for (int k=0; k<CS_WORK; k++) cpu_relax();
		my_sem_post(&sem);
	}
	return NULL;
}

// fenwick tree over arrival numbers -> bypass count per pass
struct fenwick {
	std::vector<int> t;
	fenwick (int n) : t(n+1, 0) {}
	void add (int i) { for (i++; i < (int)t.size(); i += i & -i) t[i]++; }
	int sum (int i) { int s = 0; for (i++; i > 0; i -= i & -i) s += t[i]; return s; }
};

void run_fifo (void)
{
	long M = (long)T*K;
	log_arrival = (int*)malloc(M*sizeof(int));
	assert(log_arrival != NULL);
	my_sem_init(&sem, 1);

	std::vector<pthread_t> tid(T);
	struct timeval tv0, tv1;
	gettimeofday(&tv0, NULL);
	for (int i=0; i<T; i++) pthread_create(&tid[i], NULL, fifo_worker, NULL);
	for (int i=0; i<T; i++) pthread_join(tid[i], NULL);
	gettimeofday(&tv1, NULL);
	assert(log_len == M);

	// overtaken = passes whose arrival is below an earlier pass's arrival
	// bypass(x) = later arrivals that passed before x
	long overtaken = 0, max_bypass = 0, sum_bypass = 0;
	fenwick fw(M);
	for (long i=0; i<M; i++) {
		int x = log_arrival[i];
		long bypass = i - fw.sum(x);	// earlier passes with arrival > x
		if (bypass > 0) overtaken++;
		if (bypass > max_bypass) max_bypass = bypass;
		sum_bypass += bypass;
		fw.add(x);
	}
	long us = (tv1.tv_sec-tv0.tv_sec)*1000000L+(tv1.tv_usec-tv0.tv_usec);
	printf("P%d fifo T=%d K=%d passes=%ld overtaken=%ld (%.3f%%) max_bypass=%ld avg_bypass=%.3f time_us=%ld\n",
		VERSION, T, K, M, overtaken, 100.0*overtaken/M, max_bypass, (double)sum_bypass/M, us);
	my_sem_destroy(&sem);
	free(log_arrival);
}


// fairness: per-producer finish time in the real MPSC program
struct timeval fair_t0;
long *finish_us;

void *timed_produce (void *param)
{
	produce(param);
	struct timeval t;
	gettimeofday(&t, NULL);
	finish_us[*(int*)param] = (t.tv_sec-fair_t0.tv_sec)*1000000L+(t.tv_usec-fair_t0.tv_usec);
	return NULL;
}

void run_fair (void)
{
	array = (slot_t*)aligned_alloc(CACHE_LINE, N*sizeof(slot_t));
	assert(array != NULL);
	for (int i=0; i<N; i++) { array[i].value = 0; array[i].full = 0; }
	consumed = (int*)malloc((long)n*P*sizeof(int));
	assert(consumed != NULL);
	memset(consumed, 0, (long)n*P*sizeof(int));
	finish_us = (long*)calloc(P, sizeof(long));
	my_sem_init(&sem_prod.s, N);
	my_sem_init(&sem_cons.s, 0);

	std::vector<pthread_t> tid(P);
	std::vector<int> id(P);
	for (int i=0; i<P; i++) id[i] = i;
	gettimeofday(&fair_t0, NULL);
	for (int i=0; i<P; i++) pthread_create(&tid[i], NULL, timed_produce, &id[i]);
	consume();
	for (int i=0; i<P; i++) pthread_join(tid[i], NULL);
	struct timeval t1;
	gettimeofday(&t1, NULL);
	long total = (t1.tv_sec-fair_t0.tv_sec)*1000000L+(t1.tv_usec-fair_t0.tv_usec);

	long lo = finish_us[0], hi = finish_us[0];
	for (int i=1; i<P; i++) { if (finish_us[i] < lo) lo = finish_us[i]; if (finish_us[i] > hi) hi = finish_us[i]; }
	// first finisher's share of total time: 1.0 = perfectly fair
	printf("P%d fair N=%d P=%d n=%d check=%s total_us=%ld first_finish_us=%ld last_finish_us=%ld first/total=%.3f\n",
		VERSION, N, P, n, verify() ? "OK" : "FAILED", total, lo, hi, (double)lo/total);
	my_sem_destroy(&sem_prod.s);
	my_sem_destroy(&sem_cons.s);
}

int main (int argc, char *argv[])
{
	if (argc == 4 && !strcmp(argv[1], "fifo")) {
		T = atoi(argv[2]);
		K = atoi(argv[3]);
		run_fifo();
	}
	else if (argc == 5 && !strcmp(argv[1], "fair")) {
		N = atoi(argv[2]);
		P = atoi(argv[3]);
		n = atoi(argv[4]);
		run_fair();
	}
	else {
		printf("usage: %s fifo T K | fair N P n\n", argv[0]);
		return 1;
	}
	return 0;
}
