from __future__ import annotations

import sys
import traceback


def _start() -> None:
    # Keep the import inside the guarded startup path so packaged dependency
    # errors remain visible as well.
    from local_workbench.server import main

    main()


def _pause_packaged_console() -> None:
    if not getattr(sys, "frozen", False):
        return
    try:
        input("\n按 Enter 键关闭窗口……")
    except (EOFError, KeyboardInterrupt):
        pass


def run() -> None:
    try:
        _start()
    except KeyboardInterrupt:
        return
    except SystemExit as exc:
        if exc.code in (None, 0):
            raise
        print(f"\nVideo Annotation Workbench 启动失败：\n{exc}", file=sys.stderr)
        _pause_packaged_console()
        raise SystemExit(1) from None
    except BaseException:
        print("\nVideo Annotation Workbench 启动失败：", file=sys.stderr)
        traceback.print_exc()
        _pause_packaged_console()
        raise SystemExit(1) from None


if __name__ == "__main__":
    run()
