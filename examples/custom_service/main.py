from haverscript import Reply, Request, Service, ServiceProvider, echo, model


class MyProvider(ServiceProvider):
    def ask(self, request: Request):
        assert request.contexture.model == "A"
        return Reply(
            [
                f"I reject your {len(request.prompt.split())} word prompt, and replace it with my own."
            ]
        )

    def list(self):
        return ["A"]


def connect(name: str):
    return Service(MyProvider()) | model(name)


session = connect("A") | echo()
session = session.chat("In one sentence, why is the sky blue?")
