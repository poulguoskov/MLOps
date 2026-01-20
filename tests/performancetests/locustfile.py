"""Load testing for the Clickbait Classifier API."""

from locust import HttpUser, between, task


class ClickbaitAPIUser(HttpUser):
    """Simulates a user interacting with the Clickbait Classifier API."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task(1)
    def health_check(self):
        """Check the health endpoint."""
        self.client.get("/")

    @task(5)
    def classify_single(self):
        """Classify a single headline (most common action)."""
        self.client.post(
            "/classify",
            json={"text": "You Won't BELIEVE What Happened Next!"},
        )

    @task(2)
    def classify_batch(self):
        """Classify a batch of headlines."""
        self.client.post(
            "/classify/batch",
            json={
                "texts": [
                    "Scientists discover high number of caterpillars",
                    "This ONE Trick Will Change Your Life Forever",
                    "New study reveals health benefits of exercise",
                ]
            },
        )
