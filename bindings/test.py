import ompeval

calc = ompeval.EquityCalculator()
hand1 = ompeval.CardRange("AhKh")
hand2 = ompeval.CardRange("QcQs")

calc.start([hand1, hand2], enumerateAll=True, stdevTarget=0.01)
calc.wait()
res = calc.get_results()

print("Players:", res.players)
print("Equities:", res.equity)
print("Wins:", res.wins)
print("Ties:", res.ties)
print("Hands:", res.hands)
print("Finished:", res.finished)