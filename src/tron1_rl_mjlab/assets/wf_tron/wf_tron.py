from pathlib import Path
import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

current_dir: Path = Path(__file__).parent.resolve()

WF_TRON_XML: Path = current_dir / "xml" / "robot.xml"
assert WF_TRON_XML.exists(), f"XML file not found: {WF_TRON_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(WF_TRON_XML))


WF_TRON_LEG_ACTUATORS = BuiltinPositionActuatorCfg(
    target_names_expr=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"),
    effort_limit=80.0,
    stiffness=40.0,
    damping=1.8,
)

WF_TRON_WHEEL_ACTUATORS = BuiltinVelocityActuatorCfg(
    target_names_expr=("wheel_[RL]_Joint",),
    effort_limit=40.0,
    damping=0.5,
    frictionloss=0.33,
)

WF_TRON_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        WF_TRON_LEG_ACTUATORS,
        WF_TRON_WHEEL_ACTUATORS,
    ),
)

WF_TRON_CONTACT_SENSOR = ContactSensorCfg(
    name="contact_sensors",
    primary=ContactMatch(mode="body", pattern="base_Link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern=".*"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=10,
)

WF_TRON_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8+0.166),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    articulation=WF_TRON_ARTICULATION,
)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    robot = Entity(WF_TRON_ROBOT_CFG)
    viewer.launch(robot.spec.compile())
