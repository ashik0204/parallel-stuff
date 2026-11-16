// CS433 Assignment 2, Question 1
// Group 1: Ashik Stenny, Spandan Pati, Vedhanth Balasubramanian
// multi-producer single-consumer bounded buffer
// version picked at compile time:
//   -DVERSION=1  P1: lock-free FIFO semaphore (cmpxchg only, busy-wait)
//   -DVERSION=2  P2: lock-based busy-wait semaphore (cmpxchg spin lock)
//   -DVERSION=3  P3: Linux semaphore (sem_t)
//   -DVERSION=4  P4: condition variable semaphore
// usage: ./q1_pX N P n  (N = array size, P = producers, n = values per producer)
// note: P1 busy-waits by design (FIFO, no context switch)
// note: far more threads (P+1) than cores + small N -> P1 can take minutes
// producer p -> values p*n .. p*n+n-1
// consumer = main thread, stores values, prints after timing
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <semaphore.h>

#ifndef VERSION
#define VERSION 1
#endif

#define CACHE_LINE 64

// cmpxchg primitive from class (unchanged)
unsigned char CompareAndSet(int oldVal, int newVal, int *ptr)
{
  int oldValOut;
  unsigned char result;

  asm(
		"lock cmpxchgl %4, %1 \n setzb %0"
		:"=qm"(result),  "+m" (*ptr), "=a" (oldValOut)
		:"a" (oldVal),  "r" (newVal)
		:
	);

  return result;
}

// compiler barrier, as in class code before release
// lock prefix = fence for hardware only, compiler still needs this
static inline void barrier(void) { asm volatile("":::"memory"); }
static inline int load(int *ptr) { return *(volatile int*)ptr; }
static inline void store(int *ptr, int v) { barrier(); *(volatile int*)ptr = v; barrier(); }
static inline void cpu_relax(void) { asm volatile("pause":::"memory"); }

// lock-free fetch-and-increment via CAS retry, returns old value
int FetchAndInc(int *ptr)
{
	int old;
	barrier();
	do {
		old = load(ptr);
	} while (!CompareAndSet(old, (int)((unsigned)old + 1u), ptr));
	barrier();
	return old;
}

// semaphores: my_sem_init / my_sem_wait / my_sem_post / my_sem_destroy

#if VERSION == 1
// P1: lock-free FIFO ticket semaphore
// ticket = wait() calls so far, grant = initial value + post() calls so far
// wait: take ticket t, spin till grant > t
// post: grant++
// tickets in arrival order + grant only grows -> FIFO
// value = grant - ticket, compared via wrapped difference (overflow safe)
// counters on separate cache lines (waiters CAS ticket, spin on grant)
typedef struct {
	int ticket;
	char pad0[CACHE_LINE - sizeof(int)];
	int grant;
	char pad1[CACHE_LINE - sizeof(int)];
} lf_sem_counters_t;

typedef struct {
	lf_sem_counters_t *c;
} my_sem_t;

void my_sem_init(my_sem_t *s, int value)
{
	s->c = (lf_sem_counters_t*)aligned_alloc(CACHE_LINE, sizeof(lf_sem_counters_t));
	assert(s->c != NULL);
	s->c->ticket = 0;
	s->c->grant = value;
}

void my_sem_wait(my_sem_t *s)
{
	int t = FetchAndInc(&s->c->ticket);
	int ahead;
	// spin till grant > t, back off by queue position
	while ((ahead = (int)((unsigned)t - (unsigned)load(&s->c->grant))) >= 0) {
		for (int k = 0; k <= ahead; k++) cpu_relax();
	}
	asm volatile("":::"memory");
}

void my_sem_post(my_sem_t *s)
{
	FetchAndInc(&s->c->grant);
}

void my_sem_destroy(my_sem_t *s)
{
	free(s->c);
	s->c = NULL;
}

#elif VERSION == 2
// P2: count protected by cmpxchg spin lock
// optimisation: test-and-test-and-set + exponential backoff
typedef struct {
	int lock;
	int count;
} my_sem_t;

void acquire(int *lock)
{
	int backoff = 1;
	while (1) {
		while (load(lock)) cpu_relax();
		if (CompareAndSet(0, 1, lock)) { barrier(); return; }
		// lost the race -> exponential backoff
		for (int k = 0; k < backoff; k++) cpu_relax();
		if (backoff < 256) backoff <<= 1;
	}
}

void release(int *lock)
{
	store(lock, 0);
}

void my_sem_init(my_sem_t *s, int value)
{
	s->lock = 0;
	s->count = value;
}

void my_sem_wait(my_sem_t *s)
{
	while (1) {
		// no lock till count > 0
		while (load(&s->count) <= 0) cpu_relax();
		acquire(&s->lock);
		if (s->count > 0) {
			s->count--;
			release(&s->lock);
			return;
		}
		release(&s->lock);
	}
}

void my_sem_post(my_sem_t *s)
{
	acquire(&s->lock);
	s->count++;
	release(&s->lock);
}

void my_sem_destroy(my_sem_t *s) { (void)s; }

#elif VERSION == 3
// P3: Linux semaphore
typedef sem_t my_sem_t;

void my_sem_init(my_sem_t *s, int value) { sem_init(s, 0, value); }
void my_sem_wait(my_sem_t *s) { while (sem_wait(s) != 0); }
void my_sem_post(my_sem_t *s) { sem_post(s); }
void my_sem_destroy(my_sem_t *s) { sem_destroy(s); }

#elif VERSION == 4
// P4: semaphore from condition variable
// mutex only protects cond var + count
// waiters -> skip signal if nobody sleeping
typedef struct {
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	int count;
	int waiters;
} my_sem_t;

void my_sem_init(my_sem_t *s, int value)
{
	pthread_mutex_init(&s->mutex, NULL);
	pthread_cond_init(&s->cond, NULL);
	s->count = value;
	s->waiters = 0;
}

void my_sem_wait(my_sem_t *s)
{
	pthread_mutex_lock(&s->mutex);
	while (s->count == 0) {
		s->waiters++;
		pthread_cond_wait(&s->cond, &s->mutex);
		s->waiters--;
	}
	s->count--;
	pthread_mutex_unlock(&s->mutex);
}

void my_sem_post(my_sem_t *s)
{
	pthread_mutex_lock(&s->mutex);
	s->count++;
	if (s->waiters) pthread_cond_signal(&s->cond);
	pthread_mutex_unlock(&s->mutex);
}

void my_sem_destroy(my_sem_t *s)
{
	pthread_cond_destroy(&s->cond);
	pthread_mutex_destroy(&s->mutex);
}

#else
#error "VERSION must be 1, 2, 3 or 4"
#endif

// multi-producer single-consumer program

// slot = value + full flag, one cache line each (no false sharing)
typedef struct {
	int value;
	int full;
	char pad[CACHE_LINE - 2*sizeof(int)];
} slot_t;

slot_t *array;
int N, P, n;
int *consumed;			// consumed values, printed after timing

// shared variables written while running, one cache line each
// (no false sharing with each other or with N, n, array read by all)
typedef struct {
	int v;
	char pad[CACHE_LINE - sizeof(int)];
} padded_int_t;
typedef struct {
	my_sem_t s;
	char pad[CACHE_LINE - sizeof(my_sem_t) % CACHE_LINE];
} padded_sem_t;

padded_int_t write_pointer __attribute__((aligned(CACHE_LINE)));	// next slot to claim, shared by producers
padded_sem_t sem_prod __attribute__((aligned(CACHE_LINE)));		// empty slots
padded_sem_t sem_cons __attribute__((aligned(CACHE_LINE)));		// full slots

void *produce (void *param)
{
	int id = *(int*)(param);
	int base = id*n;

	for (int i=0; i<n; i++) {
		my_sem_wait(&sem_prod.s);
		// claim slot, many producers past sem_wait -> lock-free CAS
		int idx = (int)((unsigned)FetchAndInc(&write_pointer.v) % (unsigned)N);
		array[idx].value = base + i;
		store(&array[idx].full, 1);	// publish after value
		my_sem_post(&sem_cons.s);
	}
	return NULL;
}

void consume (void)
{
	int read_pointer = 0;
	long total = (long)n*P;

	for (long i=0; i<total; i++) {
		my_sem_wait(&sem_cons.s);
		// slot may be claimed but not written yet (producers finish out of order)
		// claimed before any later slot -> short wait
		while (!load(&array[read_pointer].full)) cpu_relax();
		consumed[i] = array[read_pointer].value;
		store(&array[read_pointer].full, 0);
		my_sem_post(&sem_prod.s);
		read_pointer++;
		if (read_pointer == N) read_pointer = 0;
	}
}

// check: each value 0..nP-1 exactly once, per-producer order kept
int verify (void)
{
	long total = (long)n*P;
	char *seen = (char*)calloc(total, 1);
	int *last = (int*)malloc(P*sizeof(int));
	assert(seen != NULL && last != NULL);
	for (int p=0; p<P; p++) last[p] = -1;
	int ok = 1;
	for (long i=0; i<total && ok; i++) {
		int v = consumed[i];
		if (v < 0 || v >= total || seen[v]) { ok = 0; break; }
		seen[v] = 1;
		int p = v / n, k = v % n;
		if (k <= last[p]) ok = 0;
		last[p] = k;
	}
	free(seen);
	free(last);
	return ok;
}

int main (int argc, char *argv[])
{
	int i, *id;
	pthread_t *tid;
	pthread_attr_t attr;
	struct timeval tv0, tv1;

	if (argc != 4) {
		printf ("Need array size (N), number of producers (P), values per producer (n).\n");
		exit(1);
	}
	N = atoi(argv[1]);
	P = atoi(argv[2]);
	n = atoi(argv[3]);
	assert(N > 0 && P > 0 && n > 0);
	assert((long)n*P < (1L << 31));

	array = (slot_t*)aligned_alloc(CACHE_LINE, N*sizeof(slot_t));
	assert(array != NULL);
	for (i=0; i<N; i++) { array[i].value = 0; array[i].full = 0; }

	consumed = (int*)malloc((long)n*P*sizeof(int));
	assert(consumed != NULL);
	memset(consumed, 0, (long)n*P*sizeof(int));	// touch pages before timing

	my_sem_init(&sem_prod.s, N);
	my_sem_init(&sem_cons.s, 0);

	tid = (pthread_t*)malloc(P*sizeof(pthread_t));
	id = (int*)malloc(P*sizeof(int));
	assert(tid != NULL && id != NULL);
	for (i=0; i<P; i++) id[i] = i;

	pthread_attr_init(&attr);

	gettimeofday(&tv0, NULL);

	for (i=0; i<P; i++) {
		pthread_create(&tid[i], &attr, produce, &id[i]);
	}
	consume();
	for (i=0; i<P; i++) {
		pthread_join(tid[i], NULL);
	}

	gettimeofday(&tv1, NULL);

	long us = (tv1.tv_sec-tv0.tv_sec)*1000000L+(tv1.tv_usec-tv0.tv_usec);

	for (long j=0; j<(long)n*P; j++) printf("%d\n", consumed[j]);

	fprintf(stderr, "P%d N=%d P=%d n=%d check=%s Time: %ld microseconds\n",
		VERSION, N, P, n, verify() ? "OK" : "FAILED", us);

	my_sem_destroy(&sem_prod.s);
	my_sem_destroy(&sem_cons.s);
	free(array);
	free(consumed);
	free(tid);
	free(id);
	return 0;
}
