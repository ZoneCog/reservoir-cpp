# Functionality Tests GitHub Action

This directory contains the GitHub Action workflow for automated functionality testing that compares Python and C++ implementations to track migration progress.

## Overview

The **Functionality Tests and Issue Management** workflow (`functionality-tests.yml`) automatically:

1. **Verifies** Python to C++ functionality implementation status
2. **Creates/Updates** GitHub issues for missing implementations  
3. **Tracks** implementation progress over time
4. **Fails** CI when significant functionality is missing

## Workflow Jobs

### 1. functionality-verification
- Analyzes Python modules in `TO_REMOVE/reservoirpy/`
- Checks for corresponding C++ implementations in `include/` and `src/`
- Generates detailed reports with missing functions and classes
- Outputs verification results and missing item counts

### 2. manage-issues  
- Creates/updates GitHub issues for missing functionality
- Generates overview issue tracking total progress
- Creates individual issues for high-priority missing items
- Uses labels: `missing-implementation`, `high-priority`, `medium-priority`

### 3. report-status
- Reports workflow success/failure status
- Fails the workflow if coverage is below 90%
- Provides summary of missing functionality

## Triggers

The workflow runs on:
- **Push** to main branches (`master`, `main`, `develop`) affecting relevant files
- **Pull requests** with changes to implementation files
- **Daily schedule** at 6 AM UTC to track progress
- **Manual dispatch** for on-demand testing

## Current Status

Based on the most recent analysis:
- **Functions**: 73/102 implemented (71.6%)
- **Classes**: 3/6 implemented (50%) 
- **Overall**: 76/108 (70.4%)

## Key Missing Items

High priority missing functions:
- `vect_wrapper` (activationsfunc.py)
- `rvs`, `data_rvs` (mat_gen.py)
- Various training state and buffer management functions

Missing classes:
- `Initializer` (mat_gen.py)
- `Unsupervised`, `FrozenModel` (node.py, model.py)

## Configuration

The workflow uses:
- **Python 3.12** for analysis
- **ubuntu-22.04** runners
- **30-day** artifact retention
- **90% coverage threshold** for success

## Customization

To modify the behavior:

1. **Change coverage threshold**: Edit the `success = coverage_percentage >= 90.0` line
2. **Add high-priority functions**: Update the `high_priority_functions` list  
3. **Modify scheduling**: Change the `cron` expression
4. **Adjust issue labels**: Update the `labels` arrays in issue creation

## Files Generated

- `missing_functions.json` - List of missing functions with modules
- `missing_classes.json` - List of missing classes with modules
- GitHub issues with detailed implementation requirements

## Integration

This workflow integrates with existing CI by:
- Using existing build dependencies and environment setup
- Following established GitHub Actions patterns in the repo
- Respecting existing path-based triggers and exclusions
- Providing actionable issue tracking for development priorities