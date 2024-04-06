import argparse

from parsely.shell import LocalShell
from parsely.tools.engineering import shell as shell_tool
from parsely_anthropic import CLAUDE_OPUS, ClaudeChat, transform_tools


class Provider:
    def __init__(self, shell):
        self._shell = shell

    def __call__(self, name, args):
        return getattr(self, name)(**args)

    def shell(self, cmd):
        text, mods, errors = self._shell.exec_from_model(cmd)
        return {"text": text}


shell = LocalShell('.')
chat = ClaudeChat(CLAUDE_OPUS,
                  'Run shell commands with the provided tool as requested by the user,'
                  'all commands should be interpreted as shell commands',
                  tools=transform_tools([shell_tool]),
                  tool_provider=Provider(shell))

parser = argparse.ArgumentParser(description='Test the shell tool')
parser.add_argument('command', help='The command to run')
args = parser.parse_args()
text = chat.get_response(args.command, stop_on_tool=True)
print(text['text'])
