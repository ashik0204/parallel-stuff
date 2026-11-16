// CS433 Assignment 2, Question 2
// Group 1: Ashik Stenny, Spandan Pati, Vedhanth Balasubramanian
// directed graph = array of sorted linked lists, list u holds v for edge (u, v)
// version picked at compile time:
//   -DVERSION=1  P1: one mutex for whole graph
//   -DVERSION=2  P2: one mutex per list
//   -DVERSION=3  P3: one mutex per node (hand-over-hand)
//   -DVERSION=4  P4: one rwlock per node (hand-over-hand)
//   -DVERSION=5  P5: one cmpxchg spin lock per node (hand-over-hand)
//   -DVERSION=6  P6: optimistic locking, cmpxchg spin lock per node
// usage: ./q2_pX op_file op_count num_threads
// op file generator (same file): g++ -O2 -DGEN_OPS -o gen_graph_ops q2_graph.cpp -lpthread
//   ./gen_graph_ops op_count insert% delete% num_vertices out_file [seed]
// op_file lines: "op u v" (0 = insert, 1 = delete, 2 = search), from gen_graph_ops
// initial graph (NUM_VERTICES, NUM_EDGES) from fixed seed -> same every run
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <assert.h>
#include <vector>
#include <algorithm>

#ifndef VERSION
#define VERSION 1
#endif

#define NUM_VERTICES 10000
#define NUM_EDGES 500000
#define GRAPH_SEED 2026

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_FIND 2

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
// (else P6 could validate with stale prev->next / deleted)
static inline void barrier(void) { asm volatile("":::"memory"); }
static inline int load(int *ptr) { return *(volatile int*)ptr; }
static inline void store(int *ptr, int v) { barrier(); *(volatile int*)ptr = v; barrier(); }
static inline void cpu_relax(void) { asm volatile("pause":::"memory"); }

// cmpxchg spin lock (P5, P6), as in Q1 P2
// optimisation: test-and-test-and-set
// no backoff: per-node locks rarely contended (10000 lists)
void acquire(int *lock)
{
	while (1) {
		while (load(lock)) cpu_relax();
		if (CompareAndSet(0, 1, lock)) { barrier(); return; }
	}
}

void release(int *lock)
{
	store(lock, 0);
}

// data structures
typedef struct list_node_s {
	int v;
#if VERSION == 3
	pthread_mutex_t lock;
#elif VERSION == 4
	pthread_rwlock_t lock;
#elif VERSION == 5
	int lock;
#elif VERSION == 6
	int lock;
	unsigned char deleted;
#endif
	struct list_node_s *next;
} list_node_t;

// next pointer read while others may write it (P4 peek, P6 walk)
static inline list_node_t *load_next(list_node_t *node) { return *(list_node_t * volatile *)&node->next; }

// list[u] = sentinel node (v = -1) before sorted list of u
// never removed -> every list starts with a lock, no head pointer to protect
// one cache line per sentinel (no false sharing between lists' head locks)
typedef struct {
	list_node_t node;
	char pad[CACHE_LINE - sizeof(list_node_t) % CACHE_LINE];
} padded_head_t;
padded_head_t *list;
static inline list_node_t *head (int u) { return &list[u].node; }

#if VERSION == 1 || VERSION == 2
typedef struct {
	pthread_mutex_t lock;
	char pad[CACHE_LINE - sizeof(pthread_mutex_t) % CACHE_LINE];
} padded_mutex_t;
#endif
#if VERSION == 1
// own cache line, apart from globals read on every op
padded_mutex_t graph_lock __attribute__((aligned(CACHE_LINE))) = { PTHREAD_MUTEX_INITIALIZER, {0} };
#elif VERSION == 2
padded_mutex_t *list_lock;
#endif

typedef struct op_node_s {
	int op;
	int u;
	int v;
} op_node_t;

op_node_t *op_array = NULL;

int num_threads, op_count;

// per-thread result counts, padded (no false sharing)
typedef struct {
	long ins_ok, del_ok, find_ok;
	char pad[CACHE_LINE - 3*sizeof(long)];
} thread_stat_t;
thread_stat_t *stats;

// node pool per thread, sized to its inserts
// allocated + locks initialised before timing -> no malloc / lock init while timing
// deleted nodes not freed during run (P6 readers may still be on them)
// all pools freed at end
// padded: used written on every insert (no false sharing)
typedef struct {
	list_node_t *nodes;
	long used, size;
	char pad[CACHE_LINE - sizeof(list_node_t*) - 2*sizeof(long)];
} node_pool_t;
node_pool_t *pools;
__thread node_pool_t *my_pool;

static inline void init_node_lock (list_node_t *node)
{
#if VERSION == 3
	pthread_mutex_init(&node->lock, NULL);
#elif VERSION == 4
	pthread_rwlock_init(&node->lock, NULL);
#elif VERSION == 5
	node->lock = 0;
#elif VERSION == 6
	node->lock = 0;
	node->deleted = 0;
#else
	(void)node;
#endif
}

static inline list_node_t *new_node (int v)
{
	assert(my_pool->used < my_pool->size);
	list_node_t *node = &my_pool->nodes[my_pool->used++];
	node->v = v;
	return node;
}

// sequential ops on one list (also used under P1, P2 locks)
int insert_seq (list_node_t *head, int v)
{
	list_node_t *prev=head, *curr=head->next;
	while (curr && (curr->v < v)) {
		prev = curr;
		curr = curr->next;
	}
	if (curr && (curr->v == v)) return 0;
	list_node_t *node = new_node(v);
	node->next = curr;
	prev->next = node;
	return 1;
}

int delete_seq (list_node_t *head, int v)
{
	list_node_t *prev=head, *curr=head->next;
	while (curr && (curr->v < v)) {
		prev = curr;
		curr = curr->next;
	}
	if (curr && (curr->v == v)) {
		prev->next = curr->next;
		return 1;
	}
	return 0;
}

int find_seq (list_node_t *head, int v)
{
	list_node_t *curr=head->next;
	while (curr && (curr->v < v)) curr = curr->next;
	return (curr && (curr->v == v));
}

// parallel ops
#if VERSION == 1

int insert_parallel (int u, int v)
{
	pthread_mutex_lock(&graph_lock.lock);
	int r = insert_seq(head(u), v);
	pthread_mutex_unlock(&graph_lock.lock);
	return r;
}
int delete_parallel (int u, int v)
{
	pthread_mutex_lock(&graph_lock.lock);
	int r = delete_seq(head(u), v);
	pthread_mutex_unlock(&graph_lock.lock);
	return r;
}
int find_parallel (int u, int v)
{
	pthread_mutex_lock(&graph_lock.lock);
	int r = find_seq(head(u), v);
	pthread_mutex_unlock(&graph_lock.lock);
	return r;
}

#elif VERSION == 2

int insert_parallel (int u, int v)
{
	pthread_mutex_lock(&list_lock[u].lock);
	int r = insert_seq(head(u), v);
	pthread_mutex_unlock(&list_lock[u].lock);
	return r;
}
int delete_parallel (int u, int v)
{
	pthread_mutex_lock(&list_lock[u].lock);
	int r = delete_seq(head(u), v);
	pthread_mutex_unlock(&list_lock[u].lock);
	return r;
}
int find_parallel (int u, int v)
{
	pthread_mutex_lock(&list_lock[u].lock);
	int r = find_seq(head(u), v);
	pthread_mutex_unlock(&list_lock[u].lock);
	return r;
}

#elif VERSION == 3 || VERSION == 5
// P3, P5: hand-over-hand, exclusive lock per node
// P3 = pthread mutex, P5 = cmpxchg spin lock, same algorithm
// walk holds only prev, lock curr then release prev
// prev held -> curr can't be unlinked, nothing inserted before it
// insert: only prev locked
// delete: also lock victim (nobody inserting right after it)
#if VERSION == 3
#define LOCK(n)   pthread_mutex_lock(&(n)->lock)
#define UNLOCK(n) pthread_mutex_unlock(&(n)->lock)
#else
#define LOCK(n)   acquire(&(n)->lock)
#define UNLOCK(n) release(&(n)->lock)
#endif

// returns with prev locked, prev->v < v <= curr->v (curr may be NULL)
static inline list_node_t *walk (list_node_t *head, int v, list_node_t **currp)
{
	list_node_t *prev = head, *curr;
	LOCK(prev);
	curr = prev->next;
	while (curr && (curr->v < v)) {
		LOCK(curr);
		UNLOCK(prev);
		prev = curr;
		curr = curr->next;
	}
	*currp = curr;
	return prev;
}

int insert_parallel (int u, int v)
{
	list_node_t *curr, *prev = walk(head(u), v, &curr);
	if (curr && (curr->v == v)) {
		UNLOCK(prev);
		return 0;
	}
	list_node_t *node = new_node(v);
	node->next = curr;
	barrier();	// not needed here (readers hold prev), same as P4/P6
	prev->next = node;
	UNLOCK(prev);
	return 1;
}

int delete_parallel (int u, int v)
{
	list_node_t *curr, *prev = walk(head(u), v, &curr);
	if (curr && (curr->v == v)) {
		LOCK(curr);
		prev->next = curr->next;
		UNLOCK(curr);
		UNLOCK(prev);
		return 1;
	}
	UNLOCK(prev);
	return 0;
}

int find_parallel (int u, int v)
{
	list_node_t *curr, *prev = walk(head(u), v, &curr);
	int r = (curr && (curr->v == v));
	UNLOCK(prev);
	return r;
}

#elif VERSION == 4
// P4: hand-over-hand, rwlock per node
// search: read locks only -> searches share nodes
// insert/delete: read locks while walking, write lock only on last predecessor
// last predecessor guessed by peeking node->next (safe, nodes never freed)
// guess checked after locking
// wrong guess (read lock on last pred) -> retry with write locks all the way
#define RDLOCK(n) pthread_rwlock_rdlock(&(n)->lock)
#define WRLOCK(n) pthread_rwlock_wrlock(&(n)->lock)
#define UNLOCK(n) pthread_rwlock_unlock(&(n)->lock)

static inline int is_last_pred (list_node_t *node, int v)
{
	list_node_t *nx = load_next(node);
	return (!nx || nx->v >= v);
}

// returns with prev write-locked, or NULL (nothing held) if retry needed
static list_node_t *walk_writer (list_node_t *head, int v, list_node_t **currp, int pessimistic)
{
	list_node_t *prev = head, *curr;
	int prev_wr = pessimistic || is_last_pred(prev, v);
	if (prev_wr) WRLOCK(prev); else RDLOCK(prev);
	curr = prev->next;
	while (curr && (curr->v < v)) {
		int curr_wr = pessimistic || is_last_pred(curr, v);
		if (curr_wr) WRLOCK(curr); else RDLOCK(curr);
		UNLOCK(prev);
		prev = curr;
		prev_wr = curr_wr;
		curr = curr->next;
	}
	if (!prev_wr) {
		UNLOCK(prev);
		return NULL;
	}
	*currp = curr;
	return prev;
}

static list_node_t *walk_writer_retry (list_node_t *head, int v, list_node_t **currp)
{
	list_node_t *prev = walk_writer(head, v, currp, 0);
	if (!prev) prev = walk_writer(head, v, currp, 1);
	return prev;
}

int insert_parallel (int u, int v)
{
	list_node_t *curr, *prev = walk_writer_retry(head(u), v, &curr);
	if (curr && (curr->v == v)) {
		UNLOCK(prev);
		return 0;
	}
	list_node_t *node = new_node(v);
	node->next = curr;
	barrier();
	prev->next = node;
	UNLOCK(prev);
	return 1;
}

int delete_parallel (int u, int v)
{
	list_node_t *curr, *prev = walk_writer_retry(head(u), v, &curr);
	if (curr && (curr->v == v)) {
		WRLOCK(curr);
		prev->next = curr->next;
		UNLOCK(curr);
		UNLOCK(prev);
		return 1;
	}
	UNLOCK(prev);
	return 0;
}

int find_parallel (int u, int v)
{
	list_node_t *prev = head(u), *curr;
	RDLOCK(prev);
	curr = prev->next;
	while (curr && (curr->v < v)) {
		RDLOCK(curr);
		UNLOCK(prev);
		prev = curr;
		curr = curr->next;
	}
	int r = (curr && (curr->v == v));
	UNLOCK(prev);
	return r;
}

#elif VERSION == 6
// P6: optimistic locking
// walk without locks, lock node(s), validate (not deleted, prev->next == curr), else retry
// delete: mark first, then unlink, never freed during run
// optimisations:
//  - insert / delete that finds nothing: lock only prev (only prev->next matters)
//  - search: no locks, present only if not marked deleted
static inline list_node_t *opt_walk (list_node_t *head, int v, list_node_t **currp)
{
	list_node_t *prev = head, *curr = load_next(head);
	while (curr && (curr->v < v)) {
		prev = curr;
		curr = load_next(curr);
	}
	*currp = curr;
	return prev;
}

static inline int valid (list_node_t *prev, list_node_t *curr)
{
	return !prev->deleted && (prev->next == curr);
}

int insert_parallel (int u, int v)
{
	list_node_t *prev, *curr;
	while (1) {
		prev = opt_walk(head(u), v, &curr);
		acquire(&prev->lock);
		if (valid(prev, curr)) {
			if (curr && (curr->v == v)) {
				release(&prev->lock);
				return 0;
			}
			list_node_t *node = new_node(v);
			node->lock = 0;
			node->deleted = 0;
			node->next = curr;
			barrier();
			prev->next = node;
			release(&prev->lock);
			return 1;
		}
		release(&prev->lock);
	}
}

int delete_parallel (int u, int v)
{
	list_node_t *prev, *curr;
	while (1) {
		prev = opt_walk(head(u), v, &curr);
		acquire(&prev->lock);
		if (curr && (curr->v == v)) {
			acquire(&curr->lock);
			if (valid(prev, curr) && !curr->deleted) {
				curr->deleted = 1;
				barrier();
				prev->next = curr->next;
				release(&curr->lock);
				release(&prev->lock);
				return 1;
			}
			release(&curr->lock);
		}
		else if (valid(prev, curr)) {
			release(&prev->lock);
			return 0;
		}
		release(&prev->lock);
	}
}

static inline int load_deleted(list_node_t *node) { return *(volatile unsigned char*)&node->deleted; }

int find_parallel (int u, int v)
{
	list_node_t *curr;
	opt_walk(head(u), v, &curr);
	return (curr && (curr->v == v) && !load_deleted(curr));
}

#else
#error "VERSION must be 1..6"
#endif

// driver
static inline long chunk_begin (int id) { return ((long)op_count*id)/num_threads; }

void *work (void *param)
{
	int id = *(int*)(param);
	long i, ins_ok=0, del_ok=0, find_ok=0;
	my_pool = &pools[id];

	if (num_threads == 1) {
		for (i=0; i<op_count; i++) {
			op_node_t *o = &op_array[i];
			if (o->op == OP_INSERT) ins_ok += insert_seq(head(o->u), o->v);
			else if (o->op == OP_DELETE) del_ok += delete_seq(head(o->u), o->v);
			else find_ok += find_seq(head(o->u), o->v);
		}
	}
	else {
		for (i=chunk_begin(id); i<chunk_begin(id+1); i++) {
			op_node_t *o = &op_array[i];
			if (o->op == OP_INSERT) ins_ok += insert_parallel(o->u, o->v);
			else if (o->op == OP_DELETE) del_ok += delete_parallel(o->u, o->v);
			else find_ok += find_parallel(o->u, o->v);
		}
	}

	stats[id].ins_ok = ins_ok;
	stats[id].del_ok = del_ok;
	stats[id].find_ok = find_ok;
	return NULL;
}

// fixed initial graph from fixed seed
// each list laid out sorted in one contiguous block
list_node_t *build_graph (void)
{
	std::vector<std::vector<int> > adj(NUM_VERTICES);
	long m = 0;
	srandom(GRAPH_SEED);
	while (m < NUM_EDGES) {
		int u = random() % NUM_VERTICES;
		int v = random() % NUM_VERTICES;
		if (u == v) continue;
		adj[u].push_back(v);
		m++;
	}
	m = 0;
	for (int u=0; u<NUM_VERTICES; u++) {
		std::sort(adj[u].begin(), adj[u].end());
		adj[u].erase(std::unique(adj[u].begin(), adj[u].end()), adj[u].end());
		m += adj[u].size();
	}

	// duplicates dropped above, top up to exactly NUM_EDGES
	while (m < NUM_EDGES) {
		int u = random() % NUM_VERTICES;
		int v = random() % NUM_VERTICES;
		if (u == v || std::binary_search(adj[u].begin(), adj[u].end(), v)) continue;
		adj[u].insert(std::lower_bound(adj[u].begin(), adj[u].end(), v), v);
		m++;
	}

	list_node_t *nodes = (list_node_t*)malloc(NUM_EDGES*sizeof(list_node_t));
	assert(nodes != NULL);
	long k = 0;
	for (int u=0; u<NUM_VERTICES; u++) {
		list_node_t *prev = head(u);
		for (size_t j=0; j<adj[u].size(); j++) {
			list_node_t *node = &nodes[k++];
			node->v = adj[u][j];
			init_node_lock(node);
			prev->next = node;
			prev = node;
		}
		prev->next = NULL;
	}
	return nodes;
}

// check: every list strictly sorted
// check: edges = initial + inserted - deleted
long count_edges (int *sorted_ok)
{
	long m = 0;
	*sorted_ok = 1;
	for (int u=0; u<NUM_VERTICES; u++) {
		int last = -1;
		for (list_node_t *c = head(u)->next; c; c = c->next) {
			if (c->v <= last || c->v == u || c->v >= NUM_VERTICES) *sorted_ok = 0;
			last = c->v;
			m++;
		}
	}
	return m;
}

#ifndef GEN_OPS
int main (int argc, char *argv[])
{
	int i, *id, op, u, v;
	FILE *fp;
	pthread_t *tid;
	pthread_attr_t attr;
	struct timeval tv0, tv1;

	if (argc != 4) {
		printf ("Need file name to read operations from, number of operations, number of threads.\n");
		exit(1);
	}
	op_count = atoi(argv[2]);
	num_threads = atoi(argv[3]);
	assert(op_count > 0 && num_threads > 0);
	op_array = (op_node_t*)malloc((long)op_count*sizeof(op_node_t));
	assert(op_array != NULL);

	fp = fopen(argv[1], "r");
	assert(fp != NULL);
	for (i=0; i<op_count; i++) {
		if (fscanf(fp, "%d %d %d", &op, &u, &v) != 3) {
			printf("Operation file has fewer than %d operations.\n", op_count);
			exit(1);
		}
		assert(u >= 0 && u < NUM_VERTICES && v >= 0 && v < NUM_VERTICES);
		op_array[i].op = op;
		op_array[i].u = u;
		op_array[i].v = v;
	}
	fclose(fp);

	list = (padded_head_t*)aligned_alloc(CACHE_LINE, NUM_VERTICES*sizeof(padded_head_t));
	assert(list != NULL);
	for (u=0; u<NUM_VERTICES; u++) {
		head(u)->v = -1;
		head(u)->next = NULL;
		init_node_lock(head(u));
	}
#if VERSION == 2
	list_lock = (padded_mutex_t*)aligned_alloc(CACHE_LINE, NUM_VERTICES*sizeof(padded_mutex_t));
	assert(list_lock != NULL);
	for (u=0; u<NUM_VERTICES; u++) pthread_mutex_init(&list_lock[u].lock, NULL);
#endif
	list_node_t *initial_nodes = build_graph();

	// node pools sized to each thread's inserts
	pools = (node_pool_t*)aligned_alloc(CACHE_LINE, num_threads*sizeof(node_pool_t));
	stats = (thread_stat_t*)aligned_alloc(CACHE_LINE, num_threads*sizeof(thread_stat_t));
	assert(pools != NULL && stats != NULL);
	for (i=0; i<num_threads; i++) {
		long b = (num_threads == 1) ? 0 : chunk_begin(i);
		long e = (num_threads == 1) ? op_count : chunk_begin(i+1);
		long inserts = 0;
		for (long j=b; j<e; j++) if (op_array[j].op == OP_INSERT) inserts++;
		pools[i].size = inserts;
		pools[i].used = 0;
		pools[i].nodes = (list_node_t*)malloc((inserts > 0 ? inserts : 1)*sizeof(list_node_t));
		assert(pools[i].nodes != NULL);
		for (long j=0; j<inserts; j++) init_node_lock(&pools[i].nodes[j]);
	}

	tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
	id = (int*)malloc(num_threads*sizeof(int));
	for (i=0; i<num_threads; i++) id[i] = i;

	pthread_attr_init(&attr);

	gettimeofday(&tv0, NULL);

	for (i=1; i<num_threads; i++) {
		pthread_create(&tid[i], &attr, work, &id[i]);
	}
	work(&id[0]);

	for (i=1; i<num_threads; i++) {
		pthread_join(tid[i], NULL);
	}

	gettimeofday(&tv1, NULL);

	long us = (tv1.tv_sec-tv0.tv_sec)*1000000L+(tv1.tv_usec-tv0.tv_usec);

	long ins_ok=0, del_ok=0, find_ok=0;
	for (i=0; i<num_threads; i++) {
		ins_ok += stats[i].ins_ok;
		del_ok += stats[i].del_ok;
		find_ok += stats[i].find_ok;
	}
	int sorted_ok;
	long edges = count_edges(&sorted_ok);
	int ok = sorted_ok && (edges == NUM_EDGES + ins_ok - del_ok);

	printf("P%d threads=%d ops=%d inserted=%ld deleted=%ld found=%ld edges=%ld check=%s\n",
		VERSION, num_threads, op_count, ins_ok, del_ok, find_ok, edges, ok ? "OK" : "FAILED");
	printf("Time: %ld microseconds, throughput: %.2f ops per second\n", us, ((double)op_count*1000000.0)/us);

	for (i=0; i<num_threads; i++) free(pools[i].nodes);
	free(pools);
	free(stats);
	free(initial_nodes);
	free(list);
	free(op_array);
	free(tid);
	free(id);
	return ok ? 0 : 2;
}
#else
// op list generator, built from this same file with -DGEN_OPS
// exact op counts + Fisher-Yates shuffle -> op types uniformly mixed over whole list
int main (int argc, char **argv)
{
	int op_count, inserts, deletes, num_vertices, i, j, t, u, v;
	unsigned seed = 433;
	char *ops;
	FILE *fp;

	if (argc != 6 && argc != 7) {
		printf("Need op count, percentage of inserts, percentage of deletes, number of vertices, output file [, seed].\nAborting...\n");
		exit(0);
	}

	op_count = atoi(argv[1]);
	inserts = (int)(((long)atoi(argv[2])*op_count)/100);
	deletes = (int)(((long)atoi(argv[3])*op_count)/100);
	num_vertices = atoi(argv[4]);
	if (argc == 7) seed = (unsigned)atoi(argv[6]);
	assert(inserts + deletes <= op_count && num_vertices > 1);

	ops = (char*)malloc(op_count);
	assert(ops != NULL);
	for (i=0; i<op_count; i++) {
		if (i < inserts) ops[i] = OP_INSERT;
		else if (i < inserts + deletes) ops[i] = OP_DELETE;
		else ops[i] = OP_FIND;
	}

	srandom(seed);
	for (i=op_count-1; i>0; i--) {
		j = random() % (i+1);
		t = ops[i]; ops[i] = ops[j]; ops[j] = t;
	}

	fp = fopen(argv[5], "w");
	assert(fp != NULL);
	for (i=0; i<op_count; i++) {
		u = random() % num_vertices;
		do { v = random() % num_vertices; } while (v == u);
		fprintf(fp, "%d %d %d\n", ops[i], u, v);
	}
	fclose(fp);
	free(ops);
	return 0;
}
#endif
