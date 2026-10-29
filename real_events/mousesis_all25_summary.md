selection done mouse02
selection done mouse03
selection done mouse04
selection done mouse05
selection done mouse06
selection done mouse08
selection done mouse09
selection done mouse11
selection done mouse12
selection done mouse13
selection done mouse14
selection done mouse15
selection done mouse17
selection done mouse18
selection done mouse19
selection done mouse20
selection done mouse21
selection done mouse23
selection done mouse24
selection done mouse25
selection done mouse27
selection done mouse29
selection done mouse30
selection done mouse31
selection done mouse33
| method | clips | mice | far left | far right | mean | sel. left | sel. right | "a red car" empty | motion mask |
|---|---|---|---|---|---|---|---|---|---|
| **all 25** | | | | | | | | | |
| Event-SAMA 10 ms | 25 | 17.0 | 7.5 | 4.6 | 9.7 | 32% | 13% | 58% | 20.7 |
| Event-SAMA 75 ms | 25 | 31.2 | 12.1 | 9.7 | 17.7 | 41% | 25% | 65% | 24.4 |
| Event-SAMA 150 ms | 25 | 32.5 | 14.7 | 10.2 | 19.1 | 52% | 30% | 66% | 24.2 |
| Event-SAMA 300 ms | 25 | 32.8 | 13.6 | 10.6 | 19.0 | 53% | 29% | 55% | 23.5 |
| pseudo-RGB 75 ms -> RGB SAMA | 25 | 36.0 | 13.9 | 13.8 | 21.3 | 27% | 20% | 84% | 24.4 |
| E2VID -> RGB SAMA | 25 | 27.6 | 14.8 | 17.0 | 19.8 | 39% | 29% | 79% | 24.4 |
| Event-SAMA, per-clip density rule | 25 | 31.7 | 13.3 | 10.5 | 18.5 | 43% | 24% | 63% | 24.5 |
| **normal** | | | | | | | | | |
| Event-SAMA 10 ms | 20 | 20.1 | 7.5 | 5.8 | 11.1 | 33% | 16% | 62% | 25.0 |
| Event-SAMA 75 ms | 20 | 35.9 | 11.9 | 11.6 | 19.8 | 41% | 28% | 76% | 29.9 |
| Event-SAMA 150 ms | 20 | 38.1 | 15.6 | 12.6 | 22.1 | 53% | 33% | 71% | 29.8 |
| Event-SAMA 300 ms | 20 | 38.4 | 15.2 | 13.1 | 22.2 | 50% | 32% | 60% | 28.9 |
| pseudo-RGB 75 ms -> RGB SAMA | 20 | 44.2 | 17.1 | 17.3 | 26.2 | 31% | 25% | 80% | 29.9 |
| E2VID -> RGB SAMA | 20 | 32.5 | 16.3 | 21.3 | 23.4 | 43% | 36% | 76% | 29.9 |
| Event-SAMA, per-clip density rule | 20 | 38.5 | 14.8 | 13.1 | 22.1 | 46% | 30% | 69% | 29.8 |
| **noisy** | | | | | | | | | |
| Event-SAMA 10 ms | 5 | 4.7 | 7.4 | 0.1 | 4.1 | 30% | 2% | 42% | 3.4 |
| Event-SAMA 75 ms | 5 | 12.6 | 13.2 | 2.1 | 9.3 | 43% | 14% | 20% | 2.0 |
| Event-SAMA 150 ms | 5 | 9.8 | 10.8 | 0.3 | 7.0 | 49% | 19% | 48% | 1.9 |
| Event-SAMA 300 ms | 5 | 10.3 | 7.3 | 0.8 | 6.2 | 62% | 17% | 36% | 1.9 |
| pseudo-RGB 75 ms -> RGB SAMA | 5 | 3.2 | 1.0 | 0.0 | 1.4 | 10% | 0% | 100% | 2.0 |
| E2VID -> RGB SAMA | 5 | 7.9 | 8.9 | 0.0 | 5.6 | 23% | 0% | 94% | 2.0 |
| Event-SAMA, per-clip density rule | 5 | 4.7 | 7.4 | 0.1 | 4.1 | 30% | 2% | 42% | 3.4 |

per-clip density rule (clip: density ev/px/s -> ideal window ms -> used):
mouse02: 10.55->11->10, mouse03: 10.71->11->10, mouse04: 10.47->11->10, mouse05: 10.53->11->10, mouse06: 10.56->11->10, mouse08: 0.23->511->300, mouse09: 0.56->212->300, mouse11: 1.18->101->75, mouse12: 1.56->76->75, mouse13: 0.42->285->300, mouse14: 0.61->196->150, mouse15: 0.72->164->150, mouse17: 0.23->511->300, mouse18: 0.28->425->300, mouse19: 1.31->91->75, mouse20: 1.08->111->150, mouse21: 0.87->137->150, mouse23: 1.17->101->75, mouse24: 0.94->127->150, mouse25: 1.69->71->75, mouse27: 0.89->133->150, mouse29: 0.48->248->300, mouse30: 0.20->603->300, mouse31: 0.77->155->150, mouse33: 0.84->142->150
