from pydantic_settings import BaseSettings, CliApp


class ArgsParser(BaseSettings, cli_implicit_flags=True, cli_kebab_case=True):
    # handy shortcut
    @classmethod
    def parse_args(cls):
        return CliApp.run(cls)
