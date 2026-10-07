"""Core command handler for SPARK."""

import sys

SPARK_NAME = "SPARK"
SPARK_FULL_NAME = "Smart Personalized Automation & Responsive Knowledge"
USER_NAME = "User"


def interpret(command: str) -> str:
    """Return SPARK's response to a supported command."""
    command = command.strip().lower()

    if command == "spark activate":
        return f"{SPARK_NAME} online. Systems calibrated."

    if command == "spark identify":
        return f"I am {SPARK_NAME}, {SPARK_FULL_NAME}."

    if command == "spark status":
        return "All systems stable and awaiting your instruction."

    if command == "spark reboot":
        return "Rebooting core systems. Stand by."

    if command == "spark shutdown":
        return "Shutting down non-essential systems."

    if command == "spark diagnostics":
        return "Running diagnostics. CPU stable. Memory stable. No anomalies detected."

    if command == "spark protocols":
        return (
            "Available protocols: activate, identify, status, reboot, shutdown, "
            "diagnostics."
        )

    if command == "spark personality":
        return "Personality module active. Tone: analytical with mild sass."

    if command == "spark version":
        return "SPARK Core v1.0 — prototype build."

    if command == "spark intruder protocol":
        return "Intruder protocol simulated. Lights flashing. Alarm active."

    if command == "spark allow entry":
        return "Entry authorized. No security actions triggered."

    if command == "spark tornado exception":
        return "Tornado exception recognized. Entry allowed."

    return "Unrecognized command. Use SPARK-specific protocols."


if __name__ == "__main__":
    command = " ".join(sys.argv[1:]) or input("SPARK> ")
    print(interpret(command))
