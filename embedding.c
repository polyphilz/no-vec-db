#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TOKENS 8191

typedef struct {
    double *embedding;
    int index;
} OpenAIEmbedding;

typedef struct {
    int count;
    int max_tokens;
    char model_name[256];
    OpenAIEmbedding embedding;
} EmbeddingContainer;

double generate_random() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

void generate_embedding_vector(double *embedding, int num_dimensions) {
    for (int i = 0; i < num_dimensions; i++) {
        embedding[i] = generate_random();
    }
}

int generate_tokens(int max_tokens) {
    return rand() % max_tokens + 1;
}

void init_embedding_container(EmbeddingContainer *container, int num_dimensions, int max_tokens, const char *model_name, int index) {
    container->count = 1;
    container->max_tokens = max_tokens;
    strncpy(container->model_name, model_name, sizeof(container->model_name));
    generate_embedding_vector(container->embedding.embedding, num_dimensions);
    container->embedding.index = index;
}

OpenAIEmbedding *to_dict(EmbeddingContainer *container) {
    return &container->embedding;
}

void save_to_file(EmbeddingContainer *container, const char *file_path) {
    FILE *file = fopen(file_path, "wb");
    if (file == NULL) {
        return;
    }
    fprintf(file, "{\"data\":[{\"embedding\":[");
    for (int i = 0; i < container->count; i++) {
        fprintf(file, "%f,", container->embedding.embedding[i]);
    }
    fprintf(file, "],\"index\":%d,\"object\":\"embedding\"}],\"model\":\"%s\",\"object\":\"list\",\"usage\":{\"prompt_tokens\":%d,\"total_tokens\":%d}}", container->embedding.index, container->model_name, container->max_tokens, container->max_tokens);
    fclose(file);
}
