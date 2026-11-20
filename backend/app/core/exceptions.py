class ApplicationError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.message = message
        self.status_code = status_code


class ServiceUnavailableError(ApplicationError):
    def __init__(self, service_name: str) -> None:
        super().__init__(
            code="service_unavailable",
            message=(
                f"{service_name} servisine geçici olarak "
                f"erişilemiyor. Lütfen daha sonra tekrar deneyin."
            ),
            status_code=503,
        )
