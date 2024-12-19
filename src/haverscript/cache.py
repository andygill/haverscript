import sqlite3
import json
from dataclasses import asdict, dataclass, field, fields
from abc import abstractmethod
from .languagemodel import Exchange

SQL_VERSION = 2

SQL_SCHEMA = f"""
BEGIN;

PRAGMA user_version = {SQL_VERSION};

CREATE TABLE IF NOT EXISTS string_pool (
    id INTEGER PRIMARY KEY,
    string TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS string_index ON string_pool(string);

CREATE TABLE IF NOT EXISTS context (
    id INTEGER PRIMARY KEY,
    prompt INTEGER  NOT NULL,       -- what was said to the LLM
    images INTEGER NOT NULL,        -- string of list of images
    reply INTEGER NOT NULL,         -- reply from the LLM
    context INTEGER,                
    FOREIGN KEY (prompt)        REFERENCES string_pool(id),
    FOREIGN KEY (images)        REFERENCES string_pool(id),
    FOREIGN KEY (reply)         REFERENCES string_pool(id),
    FOREIGN KEY (context)       REFERENCES interactions(id)
);

CREATE INDEX IF NOT EXISTS context_prompt_index ON context(prompt);
CREATE INDEX IF NOT EXISTS context_images_index ON context(images);
CREATE INDEX IF NOT EXISTS context_reply_index ON context(reply);
CREATE INDEX IF NOT EXISTS context_context_index ON context(context);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY,
    system INTEGER,
    context INTEGER,                
    parameters INTEGER NOT NULL,     
    FOREIGN KEY (system)        REFERENCES string_pool(id),
    FOREIGN KEY (context)       REFERENCES context(id),
    FOREIGN KEY (parameters)    REFERENCES string_pool(id)
);

CREATE INDEX IF NOT EXISTS interactions_system_index ON interactions(system);
CREATE INDEX IF NOT EXISTS interactions_context_index ON interactions(context);
CREATE INDEX IF NOT EXISTS interactions_parameters_index ON interactions(parameters);

CREATE TEMPORARY TABLE blacklist (
    id INTEGER PRIMARY KEY,
    interaction INTEGER NOT NULL,
    FOREIGN KEY (interaction)    REFERENCES interactions(id)
);

COMMIT;
"""


@dataclass(frozen=True)
class TEXT:
    id: int


@dataclass(frozen=True)
class PROMPT_REPLY:
    id: int


@dataclass(frozen=True)
class CONTEXT:
    id: int


@dataclass(frozen=True)
class INTERACTION:
    id: int


@dataclass
class DB:
    conn: sqlite3.Connection

    @abstractmethod
    def text(self, text: str) -> TEXT:
        pass

    @abstractmethod
    def context_row(
        self, prompt: TEXT, images: TEXT, reply: TEXT, context: CONTEXT
    ) -> PROMPT_REPLY:
        pass

    @abstractmethod
    def interaction_row(
        self, system: TEXT, context: CONTEXT, parameters: TEXT
    ) -> CONTEXT:
        pass

    def interaction_replies(
        self,
        system: TEXT,
        context: CONTEXT,
        prompt: TEXT | None,
        images: TEXT | None,
        parameters: TEXT,
        limit: int | None,
        blacklist: bool = False,
    ) -> dict[INTERACTION, tuple[str, list[str], str]]:

        interactions_args = {
            "system": system.id,
            "parameters": parameters.id,
        }

        context_args = {
            "context": context.id,
        }
        if prompt:
            context_args["prompt"] = prompt.id

        if images:
            context_args["images"] = images.id

        rows = self.conn.execute(
            "SELECT s1.string, s2.string, s3.string, interactions.id FROM "
            " interactions JOIN context JOIN string_pool as s1 JOIN string_pool as s2 JOIN string_pool as s3 WHERE "
            " interactions.context = context.id AND "
            " context.prompt = s1.id AND "
            " context.images = s2.id AND "
            " context.reply = s3.id AND "
            + " AND ".join(
                [
                    (
                        f"interactions.{key} IS NULL"
                        if interactions_args[key] is None
                        else f"interactions.{key} = :{key}"
                    )
                    for key in interactions_args.keys()
                ]
            )
            + " AND "
            + " AND ".join(
                [
                    (
                        f"context.{key} IS NULL"
                        if context_args[key] is None
                        else f"context.{key} = :{key}"
                    )
                    for key in context_args.keys()
                ]
            )
            + (
                " AND interactions.id NOT IN (SELECT interaction FROM blacklist) "
                if blacklist
                else ""
            )
            + (f" LIMIT {limit}" if limit else ""),
            interactions_args | context_args,
        ).fetchall()

        def decode_images(txt):
            if txt == '["foo.png"]':
                return ["foo.png"]
            assert txt == "[]"
            return []

        return {
            INTERACTION(row[3]): (row[0], decode_images(row[1]), row[2]) for row in rows
        }

    def blacklist(self, key: INTERACTION):  # stale?
        self.conn.execute("INSERT INTO blacklist (interaction) VALUES (?)", (key.id,))


@dataclass
class ReadOnly(DB):

    def text(self, text: str) -> TEXT:
        if text is None:
            return TEXT(None)
        assert isinstance(text, str), f"text={text}, expecting str"
        # Retrieve the id of the string
        if row := self.conn.execute(
            "SELECT id FROM string_pool WHERE string = ?", (text,)
        ).fetchone():
            return TEXT(row[0])  # Return the id of the string
        else:
            raise ValueError

    def context_row(
        self, prompt: TEXT, images: TEXT, reply: TEXT, context: CONTEXT
    ) -> PROMPT_REPLY:

        args = {
            "prompt": prompt.id,
            "images": images.id,
            "reply": reply.id,
            "context": context.id,
        }

        if row := self.conn.execute(
            f"SELECT id FROM context WHERE "
            + " AND ".join(
                [
                    f"{key} IS NULL" if args[key] is None else f"{key} = :{key}"
                    for key in args.keys()
                ]
            ),
            args,
        ).fetchone():
            return CONTEXT(row[0])  # Return the id of the string

        raise ValueError

    def interaction_row(
        self, system: TEXT, context: CONTEXT, parameters: TEXT
    ) -> INTERACTION:

        args = {
            "system": system.id,
            "context": context.id,
            "parameters": parameters.id,
        }

        if row := self.conn.execute(
            f"SELECT id FROM interactions WHERE "
            + " AND ".join(
                [
                    f"{key} IS NULL" if args[key] is None else f"{key} = :{key}"
                    for key in args.keys()
                ]
            ),
            args,
        ).fetchone():
            return INTERACTION(row[0])  # Return the id of the string
        raise ValueError


@dataclass
class ReadAppend(DB):

    def text(self, text: str) -> TEXT:
        try:
            return ReadOnly(self.conn).text(text)
        except ValueError:
            return TEXT(
                self.conn.execute(
                    "INSERT INTO string_pool (string) VALUES (?)", (text,)
                ).lastrowid
            )

    def context_row(
        self, prompt: TEXT, images: TEXT, reply: TEXT, context: CONTEXT
    ) -> CONTEXT:
        assert isinstance(prompt, TEXT), f"prompt : {type(prompt)}, expecting : TEXT"
        assert isinstance(images, TEXT), f"images : {type(images)}, expecting : TEXT"
        assert isinstance(reply, TEXT), f"reply : {type(reply)}, expecting : TEXT"
        assert isinstance(
            context, CONTEXT
        ), f"context : {type(context)}, expecting : CONTEXT"

        try:
            return ReadOnly(self.conn).context_row(prompt, images, reply, context)
        except ValueError:
            return CONTEXT(
                self.conn.execute(
                    "INSERT INTO context (prompt, images, reply, context) VALUES (?, ?, ?, ?)",
                    (prompt.id, images.id, reply.id, context.id),
                ).lastrowid
            )

    def interaction_row(
        self, system: TEXT, context: CONTEXT, parameters: TEXT
    ) -> INTERACTION:
        assert isinstance(system, TEXT), f"system : {type(context)}, expecting : TEXT"
        assert isinstance(
            context, CONTEXT
        ), f"context : {type(context)}, expecting : CONTEXT"
        assert isinstance(
            parameters, TEXT
        ), f"parameters : {type(context)}, expecting : TEXT"

        try:
            return ReadOnly(self.conn).interaction_row(system, context, parameters)
        except ValueError:
            interaction = INTERACTION(
                self.conn.execute(
                    "INSERT INTO interactions (system, context, parameters) VALUES (?,?,?)",
                    (system.id, context.id, parameters.id),
                ).lastrowid
            )
            # The idea here is that if you have just added a result of calling a LLM,
            # then in the same session you re-ask the question, you want a new answer.
            self.blacklist(interaction)
            return interaction


class Cache:
    connections = {}

    def __init__(self, filename: str, mode: str) -> None:
        self.version = SQL_VERSION
        self.filename = filename
        self.mode = mode

        assert mode in {"r", "a", "a+"}

        if (filename) in Cache.connections:
            self.conn = Cache.connections[filename]
        else:
            self.conn = sqlite3.connect(
                filename, check_same_thread=sqlite3.threadsafety != 3
            )
            Cache.connections[filename] = self.conn
            self.conn.executescript(SQL_SCHEMA)

        if mode in {"a", "a+"}:
            self.db = ReadAppend(self.conn)
        elif mode == "r":
            self.db = ReadOnly(self.conn)

    def context(self, context):
        if not context:
            return CONTEXT(None)

        top: Exchange = context[-1]

        prompt, images, reply = top.prompt, top.images, top.reply
        context = self.context(context[:-1])
        prompt = self.db.text(prompt)
        images = self.db.text(json.dumps(images))
        reply = self.db.text(reply)
        context = self.db.context_row(prompt, images, reply, context)

        return context

    def insert_interaction(self, system, context, prompt, images, reply, parameters):
        assert (
            prompt is not None
        ), f"should not be saving empty prompt, reply = {repr(reply)}"
        context = context + (Exchange(prompt=prompt, images=images, reply=reply),)
        context = self.context(context)
        system = self.db.text(system)
        parameters = self.db.text(json.dumps(parameters))
        self.db.interaction_row(system, context, parameters)
        self.conn.commit()

    def lookup_interactions(
        self,
        system: str,
        context: tuple,
        prompt: str | None,  # None = match any
        images: list[str] | None,  # None = match any
        parameters: dict,
        limit: int | None,
        blacklist: bool,
    ) -> dict[INTERACTION, str]:

        context = self.context(context)
        system = self.db.text(system)
        if prompt:
            prompt = self.db.text(prompt)
        if images:
            images = self.db.text(json.dumps(images))
        parameters = self.db.text(json.dumps(parameters))

        return self.db.interaction_replies(
            system, context, prompt, images, parameters, limit, blacklist
        )

    def blacklist(self, key: INTERACTION):
        self.db.blacklist(key)
        self.conn.commit()
