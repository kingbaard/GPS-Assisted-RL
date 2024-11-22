from gymnasium.envs.registration import register

register(
    id="village_safe/VillageSafe-v0",
    entry_point="village_safe.envs:VillageSafeEnv",
)
