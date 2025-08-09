from db import DB, open_db


def check_hand_id_map(db):
    """
    Helper function to test hand_id_map.
    """
    hand_id_map = db.get_hand_ids()
    print()
    print(f"hand_id_map contains {len(hand_id_map)} hands.")
    print(list(hand_id_map.items())[:5])
    print(":")
    print(list(hand_id_map.items())[-5:])
    print()

    return

def check_board_id_map(db):
    """
    Helper function to test board_id_map.
    """
    board_id_map = db.get_board_ids()
    print()
    print(f"board_id_map contains {len(board_id_map)} boards.")
    print(list(board_id_map.items())[:5])
    print(":")
    print(list(board_id_map.items())[-5:])
    print()

    for suit_pattern in range(5):
        board_id_map = db.get_board_ids(suit_pattern)
        print()
        print(f"board_id_map contains {len(board_id_map)} boards for suit_pattern {suit_pattern}.")
        print(list(board_id_map.items())[:5])
        print(":")
        print(list(board_id_map.items())[-5:])
        print()        

    return

def check_evaluations(db):
    """
    Helper function to test board_id_map.
    """
    evaluations = db.get_evaluations()
    print()
    print(f"evaluations contains {len(evaluations)} evaluations.")
    print(evaluations[:5])
    print(":")
    print(evaluations[-5:])
    print()

    return


def main():

    # Initialize DB connection
    db = open_db()

    db.truncate_table("evaluations")

    # check_hand_id_map(db)
    # check_board_id_map(db)
    check_evaluations(db)

    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()
