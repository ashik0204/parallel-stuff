// CS433 Assignment 2, Question 2: op list generator
// Group 1: Ashik Stenny, Spandan Pati, Vedhanth Balasubramanian
// based on class gen_list_ops.c
// usage: ./gen_graph_ops op_count insert% delete% num_vertices out_file [seed]
// output lines: "op u v" (0 = insert, 1 = delete, 2 = search), u != v uniform in [0, num_vertices)
// exact op counts + Fisher-Yates shuffle
// -> op types uniformly mixed over whole list
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_FIND 2

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
