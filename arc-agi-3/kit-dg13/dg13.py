# SPDX-License-Identifier: MIT-0
"""Kit DG13's policy: kit DG1's relational novelty (`dg1.DgThink`, frozen at 241797d2) over kit DG3's bar test
(`dg3.Dg3Think`, frozen at b69bd659), composed by cooperative inheritance. Both files are used byte for byte; this
class adds nothing. Method order: DgThink (plan, predict_next, _remap), then Dg3Think (choose_action, effect, the bar
test), then BS's BsThink and the rest of kit BH."""
import dg1
import dg3


class Dg13Think(dg1.DgThink, dg3.Dg3Think):
    pass
