from pathlib import Path
from api.interfaces import Persona
from yaml import safe_load, dump

class PersonaManager:
    def __init__(self, app_data_dir: Path):
        self.persona_config = app_data_dir / "persona.yaml"
        self.persona = Persona.get_empty_persona()
        if self.persona_config.exists():
            with open(self.persona_config, "r") as f:
                self.persona = Persona(**safe_load(f))

    def write(self):
        with open(self.persona_config, "w") as f: dump(self.persona, f)