#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
配列の形

node 0:state
     1:height
     2:out_root_id
     3:in_root_id
     4:source,sinkにつながっているかのflg

edge 0:flow
     1:reverse_id
     2:in_node_id
     3:out_node_id
     4:in_link_id
     5:out_link_id
     6:id
*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
  
#define NODE 5
#define EDGE 7

#define NADR(NUM, ID, EL) (NUM * EL + ID)
#define EADR(NUM, ID, EL) ((NUM * 2) * EL + ID)

//node
#define STATE 0
#define HEIGHT 1
#define OUT_ROOT 2
#define IN_ROOT 3
#define ST_FLG 4

//edge
#define FLOW 0
#define ROUTE 1
#define REVERSE 2
#define IN_NODE 3
#define IN_LINK 4
#define OUT_NODE 5
#define OUT_LINK 6

int *N_NUM;
int *E_NUM;
int *n_table;
int *e_table;
int *source;
int *sink;

void link(int e_table, int from, int to, int r_edge, int flow);
int load(FILE *fp);

__global__ void node_reset(int *n_table, int *e_table, int *source, int *sink, int *N_NUM, int *E_NUM, int *e_flg) {
  int total_id = blockDim.x * blockIdx.x + threadIdx.x;
  e_flg[total_id] = 0;
  if (total_id >= N_NUM[0]) return;
  if (total_id == source[0]) {
    n_table[NADR(N_NUM[0], total_id, STATE)] = 1;
    n_table[NADR(N_NUM[0], total_id, HEIGHT)] = 0;
    return;
  }
  else if (total_id == sink[0]) {
    n_table[NADR(N_NUM[0], total_id, STATE)] = 2;
    n_table[NADR(N_NUM[0], total_id, HEIGHT)] = 0;
    return;
  }
  else {
    if (n_table[NADR(N_NUM[0], total_id, ST_FLG)] == -1) {
      n_table[NADR(N_NUM[0], total_id, HEIGHT)] = N_NUM[0];
      n_table[NADR(N_NUM[0], total_id, STATE)] = 0;
      return;
    }
    else {
      int f_flg = n_table[NADR(N_NUM[0], total_id, ST_FLG)];
      if (e_table[EADR(E_NUM[0], f_flg, FLOW)] > 0) {
	int in_node = e_table[EADR(E_NUM[0], f_flg, IN_NODE)];
	n_table[NADR(N_NUM[0], total_id, HEIGHT)] = 1;
	if (in_node == source[0]) n_table[NADR(N_NUM[0], total_id, STATE)] = 1;
	else n_table[NADR(N_NUM[0], total_id, STATE)] = 2;
      }
      else {
	n_table[NADR(N_NUM[0], total_id, HEIGHT)] = N_NUM[0];
	n_table[NADR(N_NUM[0], total_id, STATE)] = 0;
	n_table[NADR(N_NUM[0], total_id, ST_FLG)] = -1;
      }
    }
  }
}

__global__ void trace_cu(int *n_table, int *e_table, int *flg3, int *que3, int *N_NUM, int *E_NUM, int *cnt, int *flg1, int *e_flg) {
  int node_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (node_id >= N_NUM[0]) return;

  __shared__ int n_num[1];
  n_num[0] = N_NUM[0];
  __shared__ int e_num[1];
  e_num[0] = E_NUM[0];

  if (n_table[NADR(n_num[0], node_id, HEIGHT)] != cnt[0]) return;
  int state, look_n, root, link;
  if (n_table[NADR(n_num[0], node_id, STATE)] == 1) {
    state = 1;
    look_n = OUT_NODE;
    root = OUT_ROOT;
    link = OUT_LINK;
  }
  else {
    state = 2;
    look_n = IN_NODE;
    root = IN_ROOT;
    link = IN_LINK;
  }
  int old;
  int lok_node;
  int edge_id = n_table[NADR(n_num[0], node_id, root)];
  for (;;) {
    for (;;) {
      if (edge_id == -1) return;
      if (e_table[EADR(e_num[0], edge_id, FLOW)] > 0) break;
      edge_id = e_table[EADR(e_num[0], edge_id, link)];
    }
    lok_node = e_table[EADR(e_num[0], edge_id, look_n)];
    if (n_table[NADR(n_num[0], lok_node, STATE)] == 0) {
      old = atomicExch(&e_flg[lok_node], 1);
      if (old == 0) {
	n_table[NADR(n_num[0], lok_node, STATE)] = state;
	n_table[NADR(n_num[0], lok_node, HEIGHT)] = n_table[NADR(n_num[0], node_id, HEIGHT)] + 1;
	flg1[0] = 1;
      }
    }
    else if (n_table[NADR(n_num[0], lok_node, STATE)] != state) {
      old = atomicAdd(&(flg3[0]), 1);
      que3[old] = edge_id;
    }
    edge_id = e_table[EADR(e_num[0], edge_id, link)];
  }
}

__device__ void flow_t(int *e_table, int flow, int pre_edge, int edge, int N_NUM, int E_NUM, int tag, int *e_flg) {
  atomicSub(&e_table[EADR(E_NUM, edge, FLOW)], flow);
  int reverse = e_table[EADR(E_NUM, edge, REVERSE)];
  atomicAdd(&e_table[EADR(E_NUM, reverse, FLOW)], flow);

  for (;;) {
    atomicSub(&e_table[EADR(E_NUM, pre_edge, FLOW)], flow);
    reverse = e_table[EADR(E_NUM, pre_edge, REVERSE)];
    atomicAdd(&e_table[EADR(E_NUM, reverse, FLOW)], flow);
    if (pre_edge == tag) break;
    pre_edge = e_table[EADR(E_NUM, pre_edge, ROUTE)];
  }
}

__global__ void aug_cu(int *n_table, int *e_table, int *flg1, int *que1, int *source, int *sink, int *e_flg, int *flow_sum, int *N_NUM, int *E_NUM) {
  int total_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (total_id >= flg1[0]) return;
  int tag = que1[total_id];
  int old = atomicExch(&e_flg[tag], 1);
  if (old != 0) return;
  int pre_edge = tag;

  __shared__ int n_num[1];
  n_num[0] = N_NUM[0];
  __shared__ int e_num[1];
  e_num[0] = E_NUM[0];

  int node = e_table[EADR(e_num[0], tag, IN_NODE)];
  int flow = e_table[EADR(e_num[0], tag, FLOW)];
  int root, look_n, link;
  int state = 1;
  int height = n_table[NADR(N_NUM[0], node, HEIGHT)] - 1;
  for (;;) {
    if (state == 1) {
      root = IN_ROOT;
      look_n = IN_NODE;
      link = IN_LINK;
    }
    else {
      root = OUT_ROOT;
      look_n = OUT_NODE;
      link = OUT_LINK;
    }
    int edge_id = n_table[NADR(n_num[0], node, root)];
    int flow1;
    int lok_node;
    for (;;) {
      if (edge_id == -1) return;
      flow1 = e_table[EADR(e_num[0], edge_id, FLOW)];
      lok_node = e_table[EADR(e_num[0], edge_id, look_n)];
      if (flow1 > 0) {
	if (n_table[NADR(n_num[0], lok_node, HEIGHT)] == height) {
	  if (n_table[NADR(n_num[0], lok_node, STATE)] == state) {
	    old = atomicExch(&e_flg[edge_id], 1);
	    if (old == 0) {
	      flow = ((flow > flow1) ? flow1 : flow);
	      break;
	    }
	  }
	}
      }
      edge_id = e_table[EADR(e_num[0], edge_id, link)];
    }
    if (lok_node == sink[0]) {
      flow_t(e_table, flow, pre_edge, edge_id, n_num[0], e_num[0], tag, e_flg);
      atomicAdd(&flow_sum[0], flow);
      return;
    }
    else if (lok_node == source[0]) {
      node = e_table[EADR(e_num[0], tag, OUT_NODE)];
      height = n_table[NADR(n_num[0], node, HEIGHT)] - 1;
      state = 2;
    }
    else {
      node = lok_node;
      height--;
    }
    e_table[EADR(e_num[0], edge_id, ROUTE)] = pre_edge;
    pre_edge = edge_id;
  }
}

__global__ void flg_reset(int *flg) {
  flg[0] = 0;
}

__global__ void cnt_add(int *cnt) {
  cnt[0]++;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: ./bk.exe file.inp\n");
    return 1;
  }
  FILE *fp = fopen(argv[1], "r");
  if (fp == NULL) {
    printf("Can't open file [%s]\n", argv[1]);
    return 1;
  }
  int max = load(fp);
  fclose(fp);

  if (max == -1) return 0;//ファイルの形式がおかしい時return

  clock_t start, end;//時間計測用

  //hostの変数確保
  int *flg1, *flg3;
  gpuErrchk ( cudaMallocHost(&flg1, sizeof(int)) );
  gpuErrchk ( cudaMallocHost(&flg3, sizeof(int)) );
  int *flow_sum;
  gpuErrchk ( cudaMallocHost(&flow_sum, sizeof(int)) );

  size_t C_SIZE = sizeof(int);
  size_t N_SIZE = sizeof(int) * N_NUM[0] * NODE;
  size_t E_SIZE = sizeof(int) * E_NUM[0] * EDGE * 2;

  //deviceの変数確保
  int *DN_NUM;
  int *DE_NUM;
  int *nd_table;
  int *ed_table;
  int *d_flg1, *d_flg3;
  int *d_que3;
  int *d_source, *d_sink;
  int *dflow_sum;
  int *de_flg;
  int *d_cnt;

  gpuErrchk( cudaMalloc((void**)&DN_NUM, C_SIZE) );
  gpuErrchk( cudaMalloc((void**)&DE_NUM, C_SIZE) );
  gpuErrchk( cudaMalloc((void**)&d_flg1, C_SIZE) );
  gpuErrchk( cudaMalloc((void**)&d_flg3, C_SIZE) );
  gpuErrchk( cudaMalloc((void**)&d_que3, sizeof(int) * E_NUM[0]) );
  gpuErrchk( cudaMalloc((void**)&nd_table, N_SIZE) );
  gpuErrchk( cudaMalloc((void**)&ed_table, E_SIZE) );
  gpuErrchk( cudaMalloc((void**)&d_source, C_SIZE) );
  gpuErrchk( cudaMalloc((void**)&d_sink, C_SIZE) );
  gpuErrchk( cudaMalloc((void**)&dflow_sum, C_SIZE) );
  gpuErrchk( cudaMalloc((void**)&de_flg, sizeof(int) * E_NUM[0] * 2) );
  gpuErrchk (cudaMalloc((void**)&d_cnt, sizeof(int)) );

  //deviceへのコピー
  gpuErrchk( cudaMemcpy(nd_table, n_table, N_SIZE, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(ed_table, e_table, E_SIZE, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_source, source, C_SIZE, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_sink, sink, C_SIZE, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(DN_NUM, N_NUM, C_SIZE, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(DE_NUM, E_NUM, C_SIZE, cudaMemcpyHostToDevice) );

  //計測開始
  start = clock();

  flg_reset<<<1, 1>>>(dflow_sum);//flow_sumを初期化
  for (;;) {
    flg_reset<<<1, 1>>>(d_flg3);
    flg_reset<<<1, 1>>>(d_cnt);
    node_reset<<<(N_NUM[0] / 32) + 1, 32>>>(nd_table, ed_table, d_source, d_sink, DN_NUM, DE_NUM, de_flg);
    gpuErrchk( cudaThreadSynchronize() );
    //growth stage
    for (;;) {
      flg_reset<<<1, 1>>>(d_flg1);
      cnt_add<<<1, 1>>>(d_cnt);
      gpuErrchk( cudaThreadSynchronize() );
      trace_cu<<<((N_NUM[0]) / 128) + 1, 128>>>(nd_table, ed_table, d_flg3, d_que3, DN_NUM, DE_NUM, d_cnt, d_flg1, de_flg);
      gpuErrchk( cudaThreadSynchronize() );
      gpuErrchk( cudaMemcpy(flg1, d_flg1, C_SIZE, cudaMemcpyDeviceToHost) );
      if (flg1[0] == 0) break;
    }

    //ぶつかったエッジがあったかを確認
    gpuErrchk( cudaMemcpy(flg3, d_flg3, sizeof(int), cudaMemcpyDeviceToHost) );
    if (flg3[0] == 0) break;

    //augmentation stage
    for (int i = 0; i < 3; i++) {
      gpuErrchk( cudaMemset((void**)de_flg, 0, sizeof(int) * E_NUM[0] * 2) );
      aug_cu<<<flg3[0] / 32 + 1, 32>>>(nd_table, ed_table, d_flg3, d_que3, d_source, d_sink, de_flg, dflow_sum, DN_NUM, DE_NUM);
      gpuErrchk( cudaThreadSynchronize() );
    }

    gpuErrchk( cudaMemcpy(flow_sum, dflow_sum, sizeof(int), cudaMemcpyDeviceToHost) );
    printf("current_flow : %d\n", flow_sum[0]);
  }

  printf("flow_sum : %d\n", flow_sum[0]);
  end = clock();
  //計測終了
  printf("time:%.2f[s]\n", (double)(end - start) / CLOCKS_PER_SEC);

  cudaFree(DN_NUM);
  cudaFree(DE_NUM);
  cudaFree(nd_table);
  cudaFree(ed_table);
  cudaFree(d_flg1);
  cudaFree(d_flg3);
  cudaFree(d_que3);
  cudaFree(d_source);
  cudaFree(d_sink);
  cudaFree(dflow_sum);
  cudaFree(d_cnt);
  cudaFree(de_flg);

  cudaFreeHost(flg1);
  cudaFreeHost(flg3);
  cudaFreeHost(n_table);
  cudaFreeHost(e_table);
  cudaFreeHost(source);
  cudaFreeHost(sink);
  cudaFreeHost(flow_sum);
  cudaFreeHost(N_NUM);
  cudaFreeHost(E_NUM);

  return 0;
}

void link(int *n_table, int *e_table, int from, int to, int flow, int edge_id, int *count) {
  int edge = n_table[NADR(N_NUM[0], from, OUT_ROOT)];
  for (;;) {
    if (edge == -1) break;
    if (e_table[EADR(E_NUM[0], edge, OUT_NODE)] == to) {
      e_table[EADR(E_NUM[0], edge, FLOW)] += flow;
      count[0]++;
      return;
    }
    edge = e_table[EADR(E_NUM[0], edge, OUT_LINK)];
  }
  //正のエッジ
  e_table[EADR(E_NUM[0], edge_id, REVERSE)] = edge_id + 1;
  e_table[EADR(E_NUM[0], edge_id, FLOW)] = flow;
  e_table[EADR(E_NUM[0], edge_id, IN_NODE)] = from;
  e_table[EADR(E_NUM[0], edge_id, OUT_NODE)] = to;
  e_table[EADR(E_NUM[0], edge_id, OUT_LINK)] = n_table[NADR(N_NUM[0], from, OUT_ROOT)];
  n_table[NADR(N_NUM[0], from, OUT_ROOT)] = edge_id;
  e_table[EADR(E_NUM[0], edge_id, IN_LINK)] = n_table[NADR(N_NUM[0], to, IN_ROOT)];
  n_table[NADR(N_NUM[0], to, IN_ROOT)] = edge_id;
  if (from == source[0]) {
    n_table[NADR(N_NUM[0], to, ST_FLG)] = edge_id;
  }
  if (to == sink[0]) {
    n_table[NADR(N_NUM[0], from, ST_FLG)] = edge_id;
  }

  //逆(reverse)のエッジ
  e_table[EADR(E_NUM[0], (edge_id + 1), REVERSE)] = edge_id;
  e_table[EADR(E_NUM[0], (edge_id + 1), FLOW)] = 0;
  e_table[EADR(E_NUM[0], (edge_id + 1), IN_NODE)] = to;
  e_table[EADR(E_NUM[0], (edge_id + 1), OUT_NODE)] = from;
  e_table[EADR(E_NUM[0], (edge_id + 1), OUT_LINK)] = n_table[NADR(N_NUM[0], to, OUT_ROOT)];
  n_table[NADR(N_NUM[0], to, OUT_ROOT)] = edge_id + 1;
  e_table[EADR(E_NUM[0], (edge_id + 1), IN_LINK)] = n_table[NADR(N_NUM[0], from, IN_ROOT)];
  n_table[NADR(N_NUM[0], from, IN_ROOT)] = edge_id + 1;
}

int load(FILE *fp) {
  int max = 0;
  char s1[10], s2[10];
  gpuErrchk( cudaMallocHost(&N_NUM, sizeof(int)) );
  gpuErrchk( cudaMallocHost(&E_NUM, sizeof(int)) );
  int result = fscanf(fp, "%s %s %d %d\n", s1, s2, &N_NUM[0], &E_NUM[0]);
  if (result == EOF) return -1;
  gpuErrchk( cudaMallocHost(&n_table, sizeof(int) * N_NUM[0] * NODE) );
  gpuErrchk( cudaMallocHost(&e_table, sizeof(int) * E_NUM[0] * EDGE * 2) );

  for (int i = 0; i < N_NUM[0]; i++) {
    n_table[NADR(N_NUM[0], i, OUT_ROOT)] = -1;
    n_table[NADR(N_NUM[0], i, IN_ROOT)] = -1;
    n_table[NADR(N_NUM[0], i, ST_FLG)] = -1;
  }
  for (int i = 0; i < E_NUM[0] * 2; i++) {
    e_table[EADR(E_NUM[0], i, REVERSE)] = -1;
    e_table[EADR(E_NUM[0], i, IN_NODE)] = -1;
    e_table[EADR(E_NUM[0], i, OUT_NODE)] = -1;
    e_table[EADR(E_NUM[0], i, IN_LINK)] = -1;
    e_table[EADR(E_NUM[0], i, OUT_LINK)] = -1;
  }
  int t;
  result = fscanf(fp, "%s %d %s\n", s1, &t, s2);
  if (result == EOF) return -1;
  cudaMallocHost(&source, sizeof(int));
  source[0] = t;
  result = fscanf(fp, "%s %d %s\n", s1, &t, s2);
  if (result == EOF) return -1;
  cudaMallocHost(&sink, sizeof(int));
  sink[0] = t;

  int *count;
  count = new int[1]();
  for (int i = 0; i < E_NUM[0]; i++) {
    int from, to, flow;
    result = fscanf(fp, "%s %d %d %d\n", s1, &from, &to, &flow);
    if (result == EOF) break;
    if (flow > max) max = flow;
    link(n_table, e_table, from, to, flow, (i - count[0]) * 2, count);
  }
  delete[] count;

  return max;
}
