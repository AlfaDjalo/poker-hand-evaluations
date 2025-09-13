# React Components
## `Card.jsx`
- **Purpose**: Display a single card
- **Props**:
  - `card` (string | null) – e.g. "AS". null = empty slot.
  - `hidden` (boolean, default: false) – If true, renders the back of the card. ***
  - `scale` (float, default: 1.0) – Controls card size. ***
  - `isClickable` (boolean, default: true) – Disables click if false.
  - `isSelected` (boolean, default: false) – Indicator of if card has been selected.
  - `onClick` (function, optional) – Handler when card is clicked.
- **State**:
  - Nil
- **Output**: Renders an <img> card face, a styled back, or an empty slot.

## `Hand.jsx`
- **Purpose**: Display a player’s hand of N cards (face-up or face-down).
- **Props**:
  - `cards` (Array<{ card: string, hidden: boolean }> required) – Cards in the hand.
  - `scale` (float, default: 0.7) – Controls card scaling.
  - `overlap` (boolean, default: true) – Whether cards overlap horizontally.
  - `overlapOffset` (number, default: 20) – Pixel offset for overlapping.
- **State**:
  - Nil
- **Output**: Renders a row of PokerCards with optional overlap.
- **Comments**: Need to reverse the overlap so that cards on left are on top.

## `PlayerSeat.jsx`
- **Purpose**: Show one player’s seat at the table.
- **Props**:
  - `seatNumber` (number, required) – Seat index at the table.
  - `player` ({ name: string, chips: number, hand: Array<{ card: string, hidden: boolean }> } | null) – Player data or null if empty seat.
  - `isDealer` (boolean, default: false) - Whether seat is the dealer.
  - `isActive` (boolean, default: false) - Whether the seat currently has the focus.
  - `onClick` (function, optional) – Handler when seat is clicked.
- **State**:
  - Nil
- **Output**: Displays name, chips, and a PokerHand, or an “empty seat” placeholder.
- **Comments**: Need to set a maximum width for the component.

## `BoardArea.jsx`
- **Purpose**: Display communal cards.
- **Props**:
  - `boardCards` (Array<{ card: string, hidden: boolean }> required) – Cards on the boand.
  - `onClick` (function, optional) – Handler when the board area is clicked.
- **State**:
  - Nil
- **Output**: Renders board area.

## `PokerTable.jsx`
- **Purpose**: Display the poker table with board + seats.
- **Props**:
  - Nil
- **State**:
  - `players` (Record<number, PlayerData>) – Maps seat numbers to player info.
  - `boardCards` (Array<{ card: string, hidden: boolean }> planned) – Cards on the board.
  - `dealerSeat` (integer) The seam number of the dealer.
  - `activeTarget` (boolean, default: false) - Which element currently has the focus.
  - `onSeatClick` (function, optional) – Handler when a seat is clicked.
  - `onBoardClick` (function, optional) – Handler when board is clicked.
- **Output**: Renders board area and up to 6 PlayerSeats around the table.

## `CardSelector.jsx`
- **Purpose**: Let user select cards from a full deck.
- **Props**:
  - `onSubmit(modelConfig)`?
- **State**:
  - `availableCards` (string[]) – Cards left in the deck.
  - `selectedcards` (string[] or Set<string>) – Cards chosen by the user.
- **Output**: Displays deck grid, hides used cards, and shows selected cards separately.

## `EquityCalculator.jsx`
- **Purpose**: Manages inputs for equity calculation.
- **Props**:
  - Nil
- **State**:
  - Nil.
- **Output**: Fetches equity calculation from backend and displays results.

