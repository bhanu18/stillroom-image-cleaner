import argparse
import json
import subprocess
from pathlib import Path

from .config import Settings


def main():
    parser = argparse.ArgumentParser(description="Private native image-cleaning workspace")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, help="Backend port when a development proxy serves the public origin")
    sub.add_parser("worker")
    sub.add_parser("session-key")
    sub.add_parser("capabilities")
    sub.add_parser("provision-qdrant")
    sub.add_parser("rebuild-index")
    b = sub.add_parser("backup")
    b.add_argument("destination")
    r = sub.add_parser("restore")
    r.add_argument("source")
    r.add_argument("destination")
    p = sub.add_parser("provision")
    p.add_argument("model", choices=["siglip", "birefnet", "sam2", "u2net"])
    p.add_argument("--revision")
    p.add_argument("--approve-reviewed-code", action="store_true")
    a = sub.add_parser("qualify")
    a.add_argument("model", choices=["siglip", "birefnet", "sam2", "u2net"])
    a.add_argument("--fixtures", required=True)
    a.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    settings = Settings.env()
    if args.command == "serve":
        from contextlib import asynccontextmanager
        from urllib.parse import urlparse

        import uvicorn

        from .api import create_app
        from .qdrant_local import start_if_provisioned
        from .supervisor import Supervisor

        app = create_app(settings)

        @asynccontextmanager
        async def lifespan(application):
            qdrant = start_if_provisioned(settings)
            supervisor = Supervisor()
            supervisor.start()
            try:
                yield
            finally:
                supervisor.stop()
                if qdrant:
                    qdrant.terminate()
                    try:
                        qdrant.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        qdrant.kill()
                        qdrant.wait()

        app.router.lifespan_context = lifespan
        print("Local session key: run image-cleaning session-key in another terminal.", flush=True)
        address = urlparse(settings.origin)
        uvicorn.run(app, host=address.hostname, port=args.port or address.port or 8000, access_log=False)
    elif args.command == "provision-qdrant":
        from .qdrant_local import provision

        print(provision(settings.data))
    elif args.command == "worker":
        from .worker import Worker

        Worker(settings).run()
    elif args.command == "session-key":
        path = settings.data / "bootstrap.secret"
        if not path.exists():
            parser.error("Start the server once to create the local session key")
        print(path.read_text())
    elif args.command == "capabilities":
        from .models import Registry

        print(json.dumps(Registry(settings.data).capabilities(), indent=2))
    elif args.command == "provision":
        from .provision import provision, provision_u2net

        if args.model == "u2net":
            result = provision_u2net(settings.data)
        else:
            revisions = {
                "siglip": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
                "birefnet": "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
                "sam2": "de431c4043854a71d8101e17995dfe596bf101a5",
            }
            result = provision(
                settings.data, args.model, args.revision or revisions[args.model], args.approve_reviewed_code
            )
        print(json.dumps(result, indent=2))
    elif args.command == "qualify":
        from .benchmark import qualify

        print(json.dumps(qualify(settings.data, args.model, Path(args.fixtures), args.device), indent=2))
    else:
        from .db import Database
        from .lifecycle import backup, replay
        from .storage import Storage

        db = Database(settings.data / "library.sqlite3")
        storage = Storage(settings.data)
        if args.command == "backup":
            print(backup(db, storage, args.destination))
        elif args.command == "restore":
            import shutil

            destination = Path(args.destination)
            if destination.exists():
                parser.error("Restore destination must not exist")
            shutil.copytree(args.source, destination)
            if (settings.data / "tombstones").exists():
                shutil.copytree(settings.data / "tombstones", destination / "tombstones", dirs_exist_ok=True)
            replay(Database(destination / "library.sqlite3"), Storage(destination))
            print(f"Restored into {destination}. Reprovision models; rebuild Qdrant before enabling memory.")
        elif args.command == "rebuild-index":
            from .memory import MemoryIndex

            print(MemoryIndex(db, settings.qdrant_url).rebuild())


if __name__ == "__main__":
    main()
