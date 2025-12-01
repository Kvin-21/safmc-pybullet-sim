"""Quick sanity check for arena layout against SAFMC 2026 specs."""

import yaml


def validate_arena():
    """Validates arena config matches competition requirements."""
    print("=" * 70)
    print("SAFMC 2026 Arena Layout Validation")
    print("=" * 70)

    with open('configs/arena_layout.yaml', 'r') as f:
        config = yaml.safe_load(f)

    arena = config['arena']
    divider = config['central_divider']
    gates = config['gates']
    gate_positions = gates['positions']

    # Show arena dimensions
    dims = arena['dimensions']
    print(f"\nArena: {dims['length']}m × {dims['width']}m × {dims['height']}m")

    print(f"\nDivider: X={divider['start_x']}-{divider['end_x']}m, "
          f"Y={divider['y']}m, height={divider['height']}m")

    # Gate specs
    single = gates['single_gate']
    double = gates['double_gate']
    print(f"\nSingle gate: {single['outer_size']}m outer, {single['inner_size']}m inner")
    print(f"Double gate: {double['total_width']}m wide, {double['inner_size']}m inner")

    # Show each gate position
    print("\nGate positions:")
    for name, g in gate_positions.items():
        if g['type'] == 'single':
            print(f"  {g['label']}: X={g['x']}m, Y={g['y_center']}m")
        elif 'x1' in g:
            print(f"  {g['label']}: ({g['x1']}, {g['y1']}) to ({g['x2']}, {g['y2']}), yaw={g['yaw']}°")
        else:
            print(f"  {g['label']}: X={g['x']}m, Y={g['y_center']}m")

    # Run checks
    print("\n" + "=" * 70)
    print("Checks:")
    print("=" * 70)

    checks = [
        (dims['length'] == 40.0, "Arena 40m length"),
        (dims['width'] == 20.0, "Arena 20m width"),
        (divider['start_x'] == 8.0, "Divider starts at X=8m"),
        (divider['end_x'] == 28.0, "Divider ends at X=28m"),
        (divider['y'] == 10.0, "Divider at Y=10m"),
        (gate_positions['end']['x'] == 9.0, "END gate at X=9m"),
        (gate_positions['start']['x'] == 14.0, "START gate at X=14m"),
        (gate_positions['gate_1']['x'] == 26.0, "GATE_1 at X=26m"),
        (gate_positions['gate_2']['x'] == 23.5, "GATE_2 at X=23.5m"),
    ]

    all_ok = True
    for passed, desc in checks:
        mark = "✓" if passed else "✗"
        print(f"  {mark} {desc}")
        if not passed:
            all_ok = False

    print("\n" + "=" * 70)
    print("✓ All checks passed!" if all_ok else "✗ Some checks failed!")
    print("=" * 70)

    return all_ok


# Backward compatibility alias
validate_arena_layout = validate_arena


if __name__ == "__main__":
    validate_arena()