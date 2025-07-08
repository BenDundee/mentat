from src.configurator import Configurator
from src.interfaces import Persona

import simplejson as sj


class PersonaManager:

    def __init__(self, cfg: Configurator):
        self.config = cfg
        self.persona_location = self.config.app_data / "persona" / "persona.json"
        self.persona = self.load_persona()

    def load_persona(self) -> Persona:
        # Load from file
        if not self.persona_location.exists():
            return Persona.get_empty_persona()
        else:
            with open(self.persona_location, "r") as f:
                return Persona.model_validate_json(f.read())

    def write_persona(self, persona: Persona) -> None:
        with open(self.persona_location, "w") as f:
            sj.dump(persona.model_dump_json(), f, indent=4)


if __name__ == "__main__":
    pm = PersonaManager(Configurator())
    persona = pm.load_persona()
    print("wait")