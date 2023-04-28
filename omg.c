#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#include <time.h>
#include <json-c/json.h>

#define NUM_DIMENSIONS 1536
#define MAX_TOKENS 8191
#define NUM_RESULTS 10

typedef struct {
    double embedding[NUM_DIMENSIONS];
} OpenAIEmbedding;

typedef struct {
    OpenAIEmbedding *embeddings;
    int count;
} EmbeddingContainer;

typedef struct {
    OpenAIEmbedding *embedding;
    double similarity;
} SimilarEmbedding;

typedef struct {
    SimilarEmbedding *items;
    int count;
    int max_size;
} MaxHeap;

typedef struct {
    struct timespec start_time;
} Timer;

void start_timer(Timer *timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->start_time);
}

double end_timer(Timer *timer) {
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = (end_time.tv_sec - timer->start_time.tv_sec) * 1e9;
    elapsed_time += (end_time.tv_nsec - timer->start_time.tv_nsec);
    elapsed_time *= 1e-9;
    return elapsed_time;
}

double dot_product(double *a, double *b, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

double norm(double *a, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

double get_cosine_similarity(OpenAIEmbedding *a, OpenAIEmbedding *b) {
    return dot_product(a->embedding, b->embedding, NUM_DIMENSIONS) / (norm(a->embedding, NUM_DIMENSIONS) * norm(b->embedding, NUM_DIMENSIONS));
}

void max_heap_push(MaxHeap *heap, SimilarEmbedding item) {
    if (heap->count < heap->max_size) {
        heap->items[heap->count] = item;
        heap->count++;
    } else if (item.similarity > heap->items[0].similarity) {
        heap->items[0] = item;
    } else {
        return;
    }

    int current_index = heap->count - 1;
    while (current_index > 0) {
        int parent_index = (current_index - 1) / 2;
        if (heap->items[parent_index].similarity < heap->items[current_index].similarity) {
            SimilarEmbedding temp = heap->items[parent_index];
            heap->items[parent_index] = heap->items[current_index];
            heap->items[current_index] = temp;
            current_index = parent_index;
        } else {
            break;
        }
    }
}

void max_heapify(MaxHeap *heap, int index) {
    int left_child = 2 * index + 1;
    int right_child = 2 * index + 2;
    int largest = index;

    if (left_child < heap->count && heap->items[left_child].similarity > heap->items[largest].similarity) {
        largest = left_child;
    }
    if (right_child < heap->count && heap->items[right_child].similarity > heap->items[largest].similarity) {
        largest = right_child;
    }
    if (largest != index) {
        SimilarEmbedding temp = heap->items[largest];
        heap->items[largest] = heap->items[index];
        heap->items[index] = temp;
        max_heapify(heap, largest);
    }
}

MaxHeap *create_max_heap(int max_size) {
    MaxHeap *heap = (MaxHeap *)malloc(sizeof(MaxHeap));
    heap->items = (SimilarEmbedding *)malloc(max_size * sizeof(SimilarEmbedding));
    heap->count = 0;
    heap->max_size = max_size;
    return heap;
}

void destroy_max_heap(MaxHeap *heap) {
    free(heap->items);
    free(heap);
}

void load_embeddings(EmbeddingContainer *container) {
    DIR *dir = opendir("out");
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".json") == NULL) {
            continue;
        }
        char path[256];
        snprintf(path, 256, "out/%s", entry->d_name);
        FILE *file = fopen(path, "rb");
        if (file == NULL) {
            continue;
        }
        fseek(file, 0, SEEK_END);
        long file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        char *buffer = (char *)malloc(file_size);
        fread(buffer, 1, file_size, file);
        fclose(file);
        struct json_object *json = json_tokener_parse(buffer);
        free(buffer);
        struct json_object *data = json_object_array_get_idx(json_object_object_get(json, "data"), 0);
        struct json_object *embedding_array = json_object_object_get(data, "embedding");
        struct array_list *embedding_list = json_object_get_array(embedding_array);
        for (int i = 0; i < NUM_DIMENSIONS; i++) {
            container->embeddings[container->count].embedding[i] = json_object_get_double(json_object_array_get_idx(embedding_list, i));
        }
        container->count++;
        json_object_put(json);
    }
    closedir(dir);
}

void get_similar_embeddings(OpenAIEmbedding *query_embedding, EmbeddingContainer *container, MaxHeap *heap) {
    for (int i = 0; i < container->count; i++) {
        double similarity = get_cosine_similarity(query_embedding, &container->embeddings[i]);
        SimilarEmbedding item = { &container->embeddings[i], similarity };
        max_heap_push(heap, item);
    }
}

int main() {
    Timer timer;
    start_timer(&timer);

    EmbeddingContainer container;
    container.embeddings = (OpenAIEmbedding *)malloc(MAX_TOKENS * sizeof(OpenAIEmbedding));
    container.count = 0;
    load_embeddings(&container);

    printf("Loaded %d embeddings in %f seconds\n", container.count, end_timer(&timer));

    start_timer(&timer);

    OpenAIEmbedding query_embedding;
    for (int i = 0; i < NUM_DIMENSIONS; i++) {
        query_embedding.embedding[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    MaxHeap *heap = create_max_heap(NUM_RESULTS);
    get_similar_embeddings(&query_embedding, &container, heap);

    printf("Top %d similar embeddings:\n", heap->count);
    for (int i = 0; i < heap->count; i++) {
        printf("%f\n", heap->items[i].similarity);
    }
   
    destroy_max_heap(heap);
    free(container.embeddings);

    printf("Found top %d similar embeddings in %f seconds\n", NUM_RESULTS, end_timer(&timer));

    return 0;
}