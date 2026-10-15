"""CLI with offline commands by default; network generation is always explicit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fusion_engine.artifacts import write_json
from fusion_engine.controlled import VERSION
from fusion_engine.controlled.adapters import ConfiguredImageAdapter
from fusion_engine.controlled.catalog import ablation_matrix, from_catalog
from fusion_engine.controlled.job import load_job, validate_document
from fusion_engine.controlled.planning import prepare
from fusion_engine.controlled.schema import job_schema, review_schema
from fusion_engine.controlled.workflow import review_run, run, write_plan
from fusion_engine.errors import FusionError
from fusion_engine.providers.configuration import load_profile


def emit(data):
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def safe_new_output(path: Path, workspace: Path, *, directory: bool = False) -> Path:
    result = path.expanduser().resolve()
    roots = [(workspace / name).resolve() for name in ("work", "output", "config/controlled")]
    if not result.is_relative_to(workspace.resolve()) or not any(
        result.is_relative_to(root) for root in roots
    ):
        raise FusionError(
            "请将此命令的输出写入当前 workspace 的 work/、output/ 或 config/controlled/"
        )
    if result.exists():
        raise FusionError("输出已存在，请使用新的路径；不覆盖既有文件")
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="多物品多场景真实融合 v2：规划、六Mask、模型适配和硬验收"
    )
    root.add_argument("--version", action="version", version=VERSION)
    commands = root.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema", help="输出任务/人工验收 JSON Schema")
    schema.add_argument("--kind", choices=("job", "review"), default="job")
    schema.add_argument("--output", type=Path)
    for name in ("validate", "plan", "ablation-matrix"):
        command = commands.add_parser(name, help="离线校验/规划/消融计划；不调用模型")
        command.add_argument("job", type=Path)
        if name != "validate":
            command.add_argument("--output", type=Path, required=True)
    info = commands.add_parser("provider-info", help="读取YAML并显示脱敏模型配置/能力，不发请求")
    info.add_argument("--provider-config", type=Path, required=True)
    preview = commands.add_parser(
        "request-preview", help="离线构建真实请求摘要；不发送Base64或API请求"
    )
    preview.add_argument("job", type=Path)
    preview.add_argument("--provider-config", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("run", help="处理导入候选；只有 --generate 才发模型请求")
    execute.add_argument("job", type=Path)
    source = execute.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate", type=Path)
    source.add_argument("--generate", action="store_true")
    execute.add_argument("--provider-config", type=Path)
    review = commands.add_parser("review", help="应用显式人工验收证据；硬失败和未知不能自动合格")
    review.add_argument("run_dir", type=Path)
    review.add_argument("--candidate-id", required=True)
    review.add_argument("--evidence", type=Path, required=True)
    migrate = commands.add_parser(
        "migrate-catalog", help="从已有逐件目录初始化草案，不伪造Mask/支撑"
    )
    migrate.add_argument("--catalog", type=Path, default=Path("profiles/object_profiles.json"))
    migrate.add_argument("--background", type=Path, required=True)
    migrate.add_argument("--objects", nargs="+", required=True)
    migrate.add_argument("--reference", action="append", default=[], metavar="ID=PATH")
    migrate.add_argument("--output", type=Path, required=True)
    return root


def exit_status(status: str) -> int:
    return {
        "ACCEPTED": 0,
        "CONCEPT_ONLY": 0,
        "REJECTED": 3,
        "REVIEW_REQUIRED": 4,
        "INPUT_REQUIRED": 2,
        "FAILED": 2,
    }.get(status, 0)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "schema":
            result = job_schema() if args.kind == "job" else review_schema()
            if args.output:
                destination = safe_new_output(args.output, Path.cwd())
                write_json(destination, result)
                emit({"output": str(destination)})
            else:
                emit(result)
        elif args.command in ("validate", "plan", "ablation-matrix"):
            job = load_job(args.job)
            if args.command == "ablation-matrix":
                destination = safe_new_output(args.output, job.workspace)
                write_json(destination, ablation_matrix(job.data))
                emit({"output": str(destination), "model_calls": 0})
            else:
                prepared = prepare(job)
                if args.command == "plan":
                    destination = safe_new_output(args.output, job.workspace, directory=True)
                    write_plan(prepared, destination)
                emit(
                    {
                        "task_id": job.data["task_id"],
                        "schema_valid": True,
                        "readiness": prepared.plan["readiness"],
                        "issues": prepared.plan["issues"],
                        "model_calls": 0,
                        **({"output": str(destination)} if args.command == "plan" else {}),
                    }
                )
                return 0 if prepared.plan["readiness"] == "READY" else 2
        elif args.command == "provider-info":
            profile = load_profile(args.provider_config)
            emit(
                {
                    "configuration": profile.public_view(),
                    "capabilities": ConfiguredImageAdapter(profile).capabilities,
                    "model_calls": 0,
                }
            )
        elif args.command == "request-preview":
            job = load_job(args.job)
            prepared = prepare(job)
            adapter = ConfiguredImageAdapter(load_profile(args.provider_config))
            adapter.preflight(prepared)
            _, summary = adapter.build_request(prepared, prepared.prompt)
            destination = safe_new_output(args.output, job.workspace)
            write_json(
                destination,
                {
                    "request_summary": summary,
                    "capabilities": adapter.capabilities,
                    "readiness": prepared.plan["readiness"],
                    "execution_allowed": prepared.plan["readiness"] == "READY",
                    "model_calls": 0,
                },
            )
            emit(
                {
                    "output": str(destination),
                    "readiness": prepared.plan["readiness"],
                    "model_calls": 0,
                }
            )
        elif args.command == "run":
            result = run(
                load_job(args.job),
                candidate=args.candidate,
                generate=args.generate,
                provider_config=args.provider_config,
            )
            emit(
                {
                    "status": result["status"],
                    "run_dir": result["run_dir"],
                    "execution": result["execution"],
                    "candidate_count": len(result["candidates"]),
                }
            )
            return exit_status(result["status"])
        elif args.command == "review":
            result = review_run(args.run_dir, args.candidate_id, args.evidence)
            emit(
                {
                    "status": result["status"],
                    "run_dir": str(args.run_dir.resolve()),
                    "final": result.get("final"),
                }
            )
            return exit_status(result["status"])
        elif args.command == "migrate-catalog":
            overrides = {}
            for value in args.reference:
                oid, separator, filename = value.partition("=")
                if not separator or not filename or oid in overrides:
                    raise FusionError("--reference 必须是唯一的 ID=PATH")
                overrides[oid] = Path(filename).expanduser().resolve()
            if set(overrides) - set(args.objects):
                raise FusionError("参考覆盖项出现未选择的物体 ID")
            destination = safe_new_output(args.output, Path.cwd())
            result = from_catalog(
                Path.cwd(), args.catalog, args.background, args.objects, overrides
            )
            validate_document(result, job_schema(), "job")
            write_json(destination, result)
            emit({"output": str(destination), "status": "DRAFT_INPUT_REQUIRED", "model_calls": 0})
        return 0
    except FusionError as exc:
        emit({"error": type(exc).__name__, "message": str(exc)})
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        emit({"error": type(exc).__name__, "message": "读取/处理失败；未输出原始内容"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
