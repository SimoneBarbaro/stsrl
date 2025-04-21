"""Script to extract action sequence from the logs so we can replay a bug"""
import os
import stsrl.slaythespire as sts
from stsrl.game_encoding import StsEncodings


def main():
    print("Input Log file:")
    logfile = input()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(f"{dir_path}/../../logs/{logfile}") as f:
        logs = f.readlines()
    seed = int([l.replace("\n", "") for l in logs if "seedAsLong" in l][0].split("seedAsLong: ")[-1])
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    print("y to skip battles")
    skip = input()
    gc.skip_battles = (skip.lower() == "y")
    logs = [l.replace("\n", "") for l in logs if
            not gc.skip_battles or ("DEBUG:stsrl.gym.environments:Game context" in l and "<GameContext: {" not in l)
            ]
    action_choices = []
    with open(f"{dir_path}/../../logs/action-sequence.log", "w") as f:
        for l in logs:
            if "available actions" in l:
                action_choices = l.split("actions->")[1].replace("\n", "").replace(" {", "{").split(";")
            elif "execute action" in l:
                current_choice = l.split("action->")[1].replace("\n", "").replace(" {", "{")
                found = False

                if gc.screen_state == sts.ScreenState.BATTLE:
                    if bc is None:
                        bc = sts.BattleContext()
                        bc.init(gc)
                    print(bc)
                    actions = bc.get_available_actions()
                    print("expected available actions: ", action_choices)
                    print("available actions: ", [a.print_desc(bc) for a in actions])
                    for i in range(len(action_choices)):
                        if current_choice == action_choices[i]:
                            f.write(str(i) + "\n")
                            found = True
                            print("executing action->", actions[i].print_desc(bc))
                            actions[i].execute(bc)
                            break
                    if bc.outcome != sts.BattleOutcome.UNDECIDED:
                        bc.exit_battle(gc)
                        bc = None
                else:
                    bc = None
                    print(gc)
                    actions = gc.get_available_actions()
                    print("expected available actions: ", action_choices)
                    print("available actions: ", [a.print_desc(gc) for a in actions])
                    for i in range(len(action_choices)):
                        if current_choice == action_choices[i]:
                            f.write(str(i) + "\n")
                            found = True
                            print("executing action->", actions[i].print_desc(gc))
                            actions[i].execute(gc)
                            break
                if not found:
                    raise Exception("Choice not found for " + l)
    print(gc)
    actions = gc.get_available_actions()
    print("next available actions: ", [a.print_desc(gc) for a in actions])
    print(StsEncodings.encode_game(gc))
    if gc.screen_state == sts.ScreenState.BATTLE:
        bc = sts.BattleContext()
        bc.init(gc)
        print(bc)
        print(StsEncodings.encode_battle(gc, bc))
        print(bc.get_available_actions())
        print([a.print_desc(bc) for a in bc.get_available_actions()])


if __name__ == "__main__":
    main()
