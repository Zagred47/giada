"""Intentional exit-code test; never confuse this with a scientific failure."""
print('GIADA_EXPECTED_CLI_EXIT_7',flush=True)
raise SystemExit(7)
