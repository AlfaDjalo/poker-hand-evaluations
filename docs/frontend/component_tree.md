```mermaid
graph TD
  EquityCalculator --> PokerTable
  EquityCalculator --> CardSelector
  PokerTable --> BoardArea
  PokerTable --> PlayerSeat["PlayerSeat (6x)"]
  PlayerSeat --> PokerHand
  PokerHand --> PokerCard
  BoardArea --> PokerCard
  CardSelector --> PokerCard