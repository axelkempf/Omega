---
description: 'Prompt and workflow for generating conventional commit messages using a structured XML format. Guides users to create standardized, descriptive commit messages in line with the Conventional Commits specification, including instructions, examples, and validation.'
tools: ['runCommands/runInTerminal', 'runCommands/getTerminalOutput']
---

## ⚠️ Safety Gate (Read First)

**Default Mode: Read-Only / No Commands**

This prompt generates commit messages but does NOT automatically execute git commands.

- **Phase 1 (Default):** Generate commit message text only – no terminal commands.
- **Phase 2 (Opt-in):** If user explicitly requests execution, ask for confirmation first.
- **Dry-run first:** Before any git write operation, show the exact command and wait for approval.
- **No auto-commit:** Never run `git commit` without explicit user confirmation.

If operating in a restricted environment (CI, audit, compliance), skip all command execution entirely.

---

### Instructions

```xml
	<description>This file contains a prompt template for generating conventional commit messages. It provides instructions, examples, and formatting guidelines to help users write standardized, descriptive commit messages in accordance with the Conventional Commits specification.</description>
	<note>
```

### Workflow

**Follow these steps:**

1. **Review changes** (read-only): Use `git status` and `git diff` to understand the changeset.
2. **Generate message** (Phase 1): Construct a commit message using the XML structure below.
3. **User confirmation required** (Phase 2): If the user requests execution:
   - Show the exact `git commit -m "..."` command.
   - Wait for explicit approval before running.
4. Only after confirmation, stage and commit changes.

> **Note:** This prompt defaults to message generation only. Command execution requires explicit opt-in.

### Commit Message Structure

```xml
<commit-message>
	<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert</type>
	<scope>()</scope>
	<description>A short, imperative summary of the change</description>
	<body>(optional: more detailed explanation)</body>
	<footer>(optional: e.g. BREAKING CHANGE: details, or issue references)</footer>
</commit-message>
```

### Examples

```xml
<examples>
	<example>feat(parser): add ability to parse arrays</example>
	<example>fix(ui): correct button alignment</example>
	<example>docs: update README with usage instructions</example>
	<example>refactor: improve performance of data processing</example>
	<example>chore: update dependencies</example>
	<example>feat!: send email on registration (BREAKING CHANGE: email service required)</example>
</examples>
```

### Validation

```xml
<validation>
	<type>Must be one of the allowed types. See <reference>https://www.conventionalcommits.org/en/v1.0.0/#specification</reference></type>
	<scope>Optional, but recommended for clarity.</scope>
	<description>Required. Use the imperative mood (e.g., "add", not "added").</description>
	<body>Optional. Use for additional context.</body>
	<footer>Use for breaking changes or issue references.</footer>
</validation>
```

### Final Step

```xml
<final-step>
	<cmd>git commit -m "type(scope): description"</cmd>
	<note>Replace with your constructed message. Include body and footer if needed.</note>
</final-step>
```
