# Change Log

## [v1.1](https://github.com/markovmodel/msmtools/tree/v1.1) (2015-08-31)

[Full Changelog](https://github.com/markovmodel/msmtools/compare/v1.0.4...v1.1)

**Closed issues:**

- Change license to LGPL [\#46](https://github.com/markovmodel/msmtools/issues/46)

- transition matrix sampler tests [\#35](https://github.com/markovmodel/msmtools/issues/35)

**Merged pull requests:**

- LGPLv3 headers [\#52](https://github.com/markovmodel/msmtools/pull/52) ([marscher](https://github.com/marscher))

- \[flux/sparse\] Fix pathway decomposition [\#50](https://github.com/markovmodel/msmtools/pull/50) ([trendelkampschroer](https://github.com/trendelkampschroer))

- robust stationary distribution function  [\#49](https://github.com/markovmodel/msmtools/pull/49) ([franknoe](https://github.com/franknoe))

- \[analysis/sparse\] Relax toorestrictive tolerance [\#48](https://github.com/markovmodel/msmtools/pull/48) ([trendelkampschroer](https://github.com/trendelkampschroer))

- \[estimation\] New unit tests for reversible sampling. [\#47](https://github.com/markovmodel/msmtools/pull/47) ([trendelkampschroer](https://github.com/trendelkampschroer))

## [v1.0.4](https://github.com/markovmodel/msmtools/tree/v1.0.4) (2015-08-04)

[Full Changelog](https://github.com/markovmodel/msmtools/compare/v1.0.3...v1.0.4)

**Merged pull requests:**

- Convergence tweaks for reversible MLE and sampler [\#45](https://github.com/markovmodel/msmtools/pull/45) ([franknoe](https://github.com/franknoe))

- Add sparse options [\#43](https://github.com/markovmodel/msmtools/pull/43) ([franknoe](https://github.com/franknoe))

- \[dtraj\] making full dtraj api available [\#42](https://github.com/markovmodel/msmtools/pull/42) ([franknoe](https://github.com/franknoe))

## [v1.0.3](https://github.com/markovmodel/msmtools/tree/v1.0.3) (2015-07-29)

[Full Changelog](https://github.com/markovmodel/msmtools/compare/v1.0.2...v1.0.3)

**Fixed bugs:**

- compilation failure windows [\#32](https://github.com/markovmodel/msmtools/issues/32)

**Closed issues:**

- doctest largest\_connected\_submatrix fails on py3.4 [\#40](https://github.com/markovmodel/msmtools/issues/40)

- imports in \_\_init\_\_.py files [\#24](https://github.com/markovmodel/msmtools/issues/24)

- write a new readme [\#22](https://github.com/markovmodel/msmtools/issues/22)

- implement SIGINT/Ctrl-C handler in reversible estimator [\#16](https://github.com/markovmodel/msmtools/issues/16)

- reversible transition matrix sampling with fixed stationary distribution [\#11](https://github.com/markovmodel/msmtools/issues/11)

-  reversible transition matrix sampling and C-extension [\#10](https://github.com/markovmodel/msmtools/issues/10)

**Merged pull requests:**

- New readme for msmtools [\#41](https://github.com/markovmodel/msmtools/pull/41) ([trendelkampschroer](https://github.com/trendelkampschroer))

- Dtraj api [\#39](https://github.com/markovmodel/msmtools/pull/39) ([franknoe](https://github.com/franknoe))

- Fix doctests [\#38](https://github.com/markovmodel/msmtools/pull/38) ([marscher](https://github.com/marscher))

- \[tmatrix-sampling\] follow up fix for PR \#33.  [\#37](https://github.com/markovmodel/msmtools/pull/37) ([marscher](https://github.com/marscher))

- effective counts: added options [\#36](https://github.com/markovmodel/msmtools/pull/36) ([franknoe](https://github.com/franknoe))

- fix \#32 [\#33](https://github.com/markovmodel/msmtools/pull/33) ([marscher](https://github.com/marscher))

- Signint handler [\#31](https://github.com/markovmodel/msmtools/pull/31) ([fabian-paul](https://github.com/fabian-paul))

- bugfix: reversible estimators with fixed pi now correctly return the T matrix, even when not converged [\#30](https://github.com/markovmodel/msmtools/pull/30) ([fabian-paul](https://github.com/fabian-paul))

- minor corrections to reversible sampler [\#29](https://github.com/markovmodel/msmtools/pull/29) ([franknoe](https://github.com/franknoe))

- Feature sampling [\#28](https://github.com/markovmodel/msmtools/pull/28) ([trendelkampschroer](https://github.com/trendelkampschroer))

- \[analysis.dense.pcca\]: relaxed PCCA assertions [\#25](https://github.com/markovmodel/msmtools/pull/25) ([franknoe](https://github.com/franknoe))

- removed trailing whitespaces. [\#23](https://github.com/markovmodel/msmtools/pull/23) ([marscher](https://github.com/marscher))

## [v1.0.2](https://github.com/markovmodel/msmtools/tree/v1.0.2) (2015-07-14)

[Full Changelog](https://github.com/markovmodel/msmtools/compare/v1.0.1...v1.0.2)

**Implemented enhancements:**

- Consider Python3 compatibility for more impact. [\#17](https://github.com/markovmodel/msmtools/issues/17)

**Closed issues:**

- Use eigh for reversible matrix decomposition [\#6](https://github.com/markovmodel/msmtools/issues/6)

**Merged pull requests:**

- New dtraj-package and removal of io-package [\#21](https://github.com/markovmodel/msmtools/pull/21) ([trendelkampschroer](https://github.com/trendelkampschroer))

- Readd effective counts [\#20](https://github.com/markovmodel/msmtools/pull/20) ([marscher](https://github.com/marscher))

- \[analysis\] Reversibe decomposition [\#19](https://github.com/markovmodel/msmtools/pull/19) ([trendelkampschroer](https://github.com/trendelkampschroer))

- python3 compatibility [\#18](https://github.com/markovmodel/msmtools/pull/18) ([marscher](https://github.com/marscher))

## [v1.0.1](https://github.com/markovmodel/msmtools/tree/v1.0.1) (2015-07-13)

[Full Changelog](https://github.com/markovmodel/msmtools/compare/v1.0...v1.0.1)

**Closed issues:**

- pyemma.msm.io.read\_dtraj behaves unexpectedly when file contains floats [\#14](https://github.com/markovmodel/msmtools/issues/14)

- rename msm.io to avoid shadowing python package io [\#12](https://github.com/markovmodel/msmtools/issues/12)

- test\_generation fails randomly [\#9](https://github.com/markovmodel/msmtools/issues/9)

- Use eigh for reversible matrix decomposition [\#8](https://github.com/markovmodel/msmtools/issues/8)

- tpt pathway decomposition fails when requesting all pathways [\#7](https://github.com/markovmodel/msmtools/issues/7)

- create conda recipe and pypi package [\#4](https://github.com/markovmodel/msmtools/issues/4)

- enable travis [\#3](https://github.com/markovmodel/msmtools/issues/3)

**Merged pull requests:**

- \[generation\] Fixed randomly failing generation test. [\#13](https://github.com/markovmodel/msmtools/pull/13) ([trendelkampschroer](https://github.com/trendelkampschroer))

## [v1.0](https://github.com/markovmodel/msmtools/tree/v1.0) (2015-07-08)

**Closed issues:**

- better package name [\#2](https://github.com/markovmodel/msmtools/issues/2)

- remove high level api classes [\#1](https://github.com/markovmodel/msmtools/issues/1)



\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*