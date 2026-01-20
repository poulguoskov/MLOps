"""Load testing for BentoML Clickbait Classifier with adaptive batching."""

from locust import HttpUser, between, task


class BentoMLUser(HttpUser):
    """Simulates users hitting the BentoML service."""

    wait_time = between(0.1, 0.3)

    @task
    def classify(self):
        """Send single text - BentoML batches these together."""
        self.client.post(
            "/classify",
            json={"texts": ["You Won't BELIEVE What Happened Next!"]},
        )
