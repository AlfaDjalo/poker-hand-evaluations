```mermaid
flowchart TD
  App((App))
  App --> PokerTable
  App --> CardSelector

  PokerTable --> BoardArea
  PokerTable --> PlayerSeat1[PlayerSeat]
  PlayerSeat1 --> PokerHand
  PokerHand --> PokerCard

  CardSelector --> PokerCard

  subgraph State
    AppState[(Global State / Context?)]
    PokerTableState[(players, boardCards)]
    CardSelectorState[(availableCards, selectedCards)]
  end

  AppState --> PokerTable
  AppState --> CardSelector
  PokerTableState --> PlayerSeat1
  CardSelectorState --> CardSelector
