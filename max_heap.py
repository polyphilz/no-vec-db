import heapq


class MaxHeap:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.heap = []

    def push(self, item):
        embedding, similarity_score = item
        heapq.heappush(self.heap, (-similarity_score, embedding))

        if len(self.heap) > self.capacity:
            heapq.heappop(self.heap)

    def pop(self):
        if not self.heap:
            return None

        neg_similarity_score, embedding = heapq.heappop(self.heap)
        return (embedding, -neg_similarity_score)

    def top(self):
        if not self.heap:
            return None

        neg_similarity_score, embedding = self.heap[0]
        return (embedding, -neg_similarity_score)

    def items(self):
        return [
            (-neg_similarity_score, embedding)
            for neg_similarity_score, embedding in self.heap
        ]

    def __len__(self):
        return len(self.heap)
