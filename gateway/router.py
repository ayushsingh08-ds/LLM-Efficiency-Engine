# gateway/router.py

class RoundRobinRouter:
    def __init__(self, providers: list[str]):
        if not providers:
            raise ValueError("Providers list cannot be empty")

        self.providers = providers
        self.index = 0

    def next_provider(self) -> str:
        provider = self.providers[self.index]
        self.index = (self.index + 1) % len(self.providers)
        return provider