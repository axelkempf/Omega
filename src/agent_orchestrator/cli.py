"""Command Line Interface für den Omega Agent Orchestrator.

Dieses Modul stellt die CLI-Schnittstelle für den Agent Orchestrator bereit.
Ermöglicht Task-Submission, Workflow-Ausführung und Status-Abfragen.

Usage:
    python -m src.agent_orchestrator.cli submit "Implementiere Feature X"
    python -m src.agent_orchestrator.cli workflow feature-implementation
    python -m src.agent_orchestrator.cli status
    python -m src.agent_orchestrator.cli agents
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .context import ContextManager, ContextScope, get_context_manager
from .orchestrator import AgentOrchestrator, get_orchestrator
from .router import AgentRole
from .v2_detector import V2Detector
from .workflow import WorkflowEngine

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Konfiguriere Logging für CLI.
    
    Args:
        verbose: Wenn True, DEBUG-Level aktivieren
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_header(title: str) -> None:
    """Drucke einen formatierten Header.
    
    Args:
        title: Der Titel
    """
    width = 60
    print()
    print("=" * width)
    print(f" {title}")
    print("=" * width)
    print()


def print_result(result: dict[str, Any]) -> None:
    """Drucke ein Ergebnis formatiert.
    
    Args:
        result: Das Ergebnis-Dict
    """
    success = result.get("success", False)
    status = "✅ SUCCESS" if success else "❌ FAILED"
    
    print(f"Status: {status}")
    print(f"Agent: {result.get('agent', 'unknown')}")
    print(f"Task ID: {result.get('task_id', 'unknown')}")
    
    if result.get("duration"):
        print(f"Duration: {result['duration']:.2f}s")
    
    if result.get("error"):
        print(f"\nError: {result['error']}")
    
    if result.get("output"):
        print(f"\nOutput:\n{result['output'][:500]}...")
    
    if result.get("files_created"):
        print(f"\nFiles Created:")
        for f in result["files_created"][:10]:
            print(f"  + {f}")
    
    if result.get("files_modified"):
        print(f"\nFiles Modified:")
        for f in result["files_modified"][:10]:
            print(f"  ~ {f}")
    
    if result.get("next_steps"):
        print(f"\nNext Steps:")
        for step in result["next_steps"][:5]:
            print(f"  → {step}")


async def cmd_submit(args: argparse.Namespace) -> int:
    """Submit einen Task zur Ausführung.
    
    Args:
        args: CLI-Argumente
        
    Returns:
        Exit-Code (0 = Erfolg)
    """
    print_header("Task Submission")
    
    workspace = Path(args.workspace).resolve()
    if not workspace.exists():
        print(f"❌ Workspace not found: {workspace}")
        return 1
    
    # Orchestrator initialisieren
    orchestrator = get_orchestrator()
    orchestrator.initialize(workspace)
    
    # Task beschreibung
    description = " ".join(args.task)
    print(f"Task: {description}")
    print(f"Workspace: {workspace}")
    
    # Optionale Parameter
    current_file = None
    if args.file:
        current_file = Path(args.file)
        if not current_file.is_absolute():
            current_file = workspace / current_file
        print(f"Current File: {current_file}")
    
    # Force Agent wenn angegeben
    force_agent = None
    if args.agent:
        try:
            force_agent = AgentRole[args.agent.upper()]
            print(f"Forced Agent: {force_agent.value}")
        except KeyError:
            print(f"⚠️ Unknown agent role: {args.agent}")
    
    print()
    print("Routing task...")
    
    # Task routen und ausführen
    try:
        result = await orchestrator.submit_task(
            description=description,
            current_file=current_file,
            force_agent=force_agent,
        )
        
        print()
        print_result(result.to_context_entry() if hasattr(result, 'to_context_entry') else vars(result))
        
        return 0 if result.success else 1
        
    except Exception as e:
        print(f"\n❌ Task execution failed: {e}")
        logger.exception("Task execution error")
        return 1


async def cmd_workflow(args: argparse.Namespace) -> int:
    """Führe einen Workflow aus.
    
    Args:
        args: CLI-Argumente
        
    Returns:
        Exit-Code (0 = Erfolg)
    """
    print_header(f"Workflow: {args.workflow_name}")
    
    workspace = Path(args.workspace).resolve()
    if not workspace.exists():
        print(f"❌ Workspace not found: {workspace}")
        return 1
    
    # Workflow Engine initialisieren
    workflow_engine = WorkflowEngine(workspace)
    
    # Workflow finden
    available = workflow_engine.list_workflows()
    if args.workflow_name not in available:
        print(f"❌ Workflow not found: {args.workflow_name}")
        print(f"\nAvailable workflows:")
        for name, desc in available.items():
            print(f"  • {name}: {desc}")
        return 1
    
    print(f"Workspace: {workspace}")
    
    # Initiale Inputs parsen
    initial_inputs = {}
    if args.inputs:
        for inp in args.inputs:
            if "=" in inp:
                key, value = inp.split("=", 1)
                initial_inputs[key] = value
    
    if initial_inputs:
        print(f"Inputs: {initial_inputs}")
    
    print()
    print("Executing workflow...")
    
    try:
        results = await workflow_engine.execute(
            args.workflow_name,
            initial_inputs=initial_inputs,
        )
        
        print()
        print(f"Workflow completed with {len(results)} steps")
        
        all_success = all(r.success for r in results)
        
        for i, result in enumerate(results, 1):
            status = "✅" if result.success else "❌"
            duration = f"{result.duration_seconds:.1f}s" if result.duration_seconds else "?"
            print(f"  {i}. {status} {result.agent_name} ({duration})")
        
        return 0 if all_success else 1
        
    except Exception as e:
        print(f"\n❌ Workflow execution failed: {e}")
        logger.exception("Workflow execution error")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Zeige Orchestrator-Status.
    
    Args:
        args: CLI-Argumente
        
    Returns:
        Exit-Code (0 = Erfolg)
    """
    print_header("Orchestrator Status")
    
    workspace = Path(args.workspace).resolve()
    
    # V2 Detection
    detector = V2Detector()
    v2_active = detector.is_v2_context()
    
    print(f"Workspace: {workspace}")
    print(f"V2 Context: {'Yes' if v2_active else 'No'}")
    
    if v2_active:
        v2_info = detector.get_v2_info()
        print(f"  Crates: {', '.join(v2_info.get('crates', []))}")
        print(f"  Has Golden Tests: {v2_info.get('has_golden_tests', False)}")
    
    print()
    
    # Context Manager Status
    ctx_manager = get_context_manager()
    ctx_stats = ctx_manager.stats
    
    print("Context Manager:")
    print(f"  Scopes: {ctx_stats['scope_count']}")
    print(f"  Total Entries: {ctx_stats['total_entries']}")
    
    print()
    
    # Orchestrator Status
    orchestrator = get_orchestrator()
    orch_stats = orchestrator.stats
    
    print("Orchestrator:")
    print(f"  Tasks Processed: {orch_stats['tasks_processed']}")
    print(f"  Agents Registered: {orch_stats['agents_registered']}")
    
    if orch_stats.get("last_task_at"):
        print(f"  Last Task: {orch_stats['last_task_at']}")
    
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    """Liste alle verfügbaren Agenten.
    
    Args:
        args: CLI-Argumente
        
    Returns:
        Exit-Code (0 = Erfolg)
    """
    print_header("Available Agents")
    
    # Zeige Agent-Rollen aus AGENT_ROLES.md
    print("Agent Roles (from AGENT_ROLES.md):")
    print()
    
    roles = [
        ("ARCHITECT", "System-Design, ADRs, Architektur-Entscheidungen"),
        ("IMPLEMENTER", "Code schreiben und ändern"),
        ("REVIEWER", "Code Review und Qualitätssicherung"),
        ("TESTER", "Test-Generierung und Maintenance"),
        ("RESEARCHER", "Bibliotheks-Recherche und Dokumentations-Analyse"),
        ("DEVOPS", "CI/CD, Deployment, Infrastructure"),
        ("SAFETY_AUDITOR", "Sicherheits-Reviews und Prompt-Analyse"),
    ]
    
    for role, desc in roles:
        print(f"  • {role:15} - {desc}")
    
    print()
    print("Routing Rules:")
    print()
    
    rules = [
        ("ADR|architect", "→ ARCHITECT"),
        ("implement|code|fix|add", "→ IMPLEMENTER"),
        ("review|check|quality", "→ REVIEWER"),
        ("test|coverage", "→ TESTER"),
        ("research|evaluate|compare", "→ RESEARCHER"),
        ("ci|cd|deploy|docker", "→ DEVOPS"),
        ("security|audit|prompt", "→ SAFETY_AUDITOR"),
        ("V2 context (rust_core/)", "→ V2-aware routing"),
    ]
    
    for pattern, target in rules:
        print(f"  {pattern:30} {target}")
    
    return 0


def cmd_workflows(args: argparse.Namespace) -> int:
    """Liste alle verfügbaren Workflows.
    
    Args:
        args: CLI-Argumente
        
    Returns:
        Exit-Code (0 = Erfolg)
    """
    print_header("Available Workflows")
    
    workspace = Path(args.workspace).resolve()
    workflow_engine = WorkflowEngine(workspace)
    
    workflows = workflow_engine.list_workflows()
    
    if not workflows:
        print("No workflows found.")
        print()
        print(f"Workflow directory: {workflow_engine.workflows_dir}")
        print("Create .yaml files in this directory to define workflows.")
        return 0
    
    for name, description in workflows.items():
        print(f"  • {name}")
        print(f"    {description}")
        print()
    
    return 0


def cmd_detect_v2(args: argparse.Namespace) -> int:
    """Prüfe V2-Kontext für eine Datei oder Pfad.
    
    Args:
        args: CLI-Argumente
        
    Returns:
        Exit-Code (0 = V2, 1 = nicht V2)
    """
    print_header("V2 Context Detection")
    
    workspace = Path(args.workspace).resolve()
    detector = V2Detector()
    
    target = args.path
    if target:
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = workspace / target_path
        print(f"Target: {target_path}")
    else:
        target_path = None
        print("Target: <workspace root>")
    
    print()
    
    # V2 Kontext prüfen
    is_v2 = detector.is_v2_context(target_path)
    
    print(f"Is V2 Context: {'Yes' if is_v2 else 'No'}")
    
    if is_v2:
        v2_info = detector.get_v2_info(target_path)
        
        print()
        print("V2 Information:")
        print(f"  Domain: {v2_info.get('domain', 'unknown')}")
        
        if v2_info.get('crate'):
            print(f"  Crate: {v2_info['crate']}")
        
        if v2_info.get('crates'):
            print(f"  Available Crates: {', '.join(v2_info['crates'])}")
        
        if v2_info.get('instructions'):
            print(f"  Instructions: {len(v2_info['instructions'])} loaded")
        
        print()
        print("Relevant Plans:")
        for plan in v2_info.get("relevant_plans", [])[:5]:
            print(f"  • {plan}")
        
        return 0
    else:
        print("\nThis path is not within V2 context.")
        print("V2 paths include: rust_core/, python/bt/")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Erstelle den CLI Argument Parser.
    
    Returns:
        Konfigurierter ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="omega-orchestrator",
        description="Omega Agent Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Submit a task:
    %(prog)s submit "Implementiere Feature X"
    %(prog)s submit --agent implementer "Fix the bug in router.py"
    %(prog)s submit --file src/main.py "Add error handling"

  Run a workflow:
    %(prog)s workflow feature-implementation
    %(prog)s workflow v2-module --inputs module=execution

  Check status:
    %(prog)s status
    %(prog)s agents
    %(prog)s workflows

  V2 detection:
    %(prog)s detect-v2 rust_core/crates/types/
""",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "-w", "--workspace",
        default=".",
        help="Workspace root directory (default: current directory)",
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="<command>",
    )
    
    # Submit command
    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit a task for execution",
    )
    submit_parser.add_argument(
        "task",
        nargs="+",
        help="Task description",
    )
    submit_parser.add_argument(
        "-a", "--agent",
        help="Force specific agent role",
    )
    submit_parser.add_argument(
        "-f", "--file",
        help="Current file context",
    )
    
    # Workflow command
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Execute a predefined workflow",
    )
    workflow_parser.add_argument(
        "workflow_name",
        help="Name of the workflow to execute",
    )
    workflow_parser.add_argument(
        "-i", "--inputs",
        nargs="*",
        help="Initial inputs as key=value pairs",
    )
    
    # Status command
    subparsers.add_parser(
        "status",
        help="Show orchestrator status",
    )
    
    # Agents command
    subparsers.add_parser(
        "agents",
        help="List available agents",
    )
    
    # Workflows command
    subparsers.add_parser(
        "workflows",
        help="List available workflows",
    )
    
    # Detect V2 command
    detect_parser = subparsers.add_parser(
        "detect-v2",
        help="Check V2 context for a path",
    )
    detect_parser.add_argument(
        "path",
        nargs="?",
        help="Path to check (default: workspace root)",
    )
    
    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Async Main-Funktion.
    
    Args:
        args: Geparsete CLI-Argumente
        
    Returns:
        Exit-Code
    """
    if args.command == "submit":
        return await cmd_submit(args)
    elif args.command == "workflow":
        return await cmd_workflow(args)
    else:
        # Sync commands
        return -1  # Should not reach here


def main() -> int:
    """CLI Entry Point.
    
    Returns:
        Exit-Code
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    setup_logging(args.verbose)
    
    # Sync commands
    if args.command == "status":
        return cmd_status(args)
    elif args.command == "agents":
        return cmd_agents(args)
    elif args.command == "workflows":
        return cmd_workflows(args)
    elif args.command == "detect-v2":
        return cmd_detect_v2(args)
    
    # Async commands
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
