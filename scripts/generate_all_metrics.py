"""
Master Script: Generate All Metrics and Graphs
Runs all evaluation scripts and generates comprehensive visualizations
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and report status"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False

def main():
    print("="*70)
    print("COMPREHENSIVE METRICS AND GRAPH GENERATION")
    print("="*70)
    print("\nThis script will:")
    print("1. Evaluate all text emotion models")
    print("2. Evaluate all facial emotion models")
    print("3. Generate all report graphs")
    print("\nEstimated time: 5-10 minutes")
    print("="*70)

    scripts_dir = Path('scripts')
    results = []

    # Step 1: Evaluate text models
    script = scripts_dir / 'evaluate_text_models.py'
    if script.exists():
        success = run_script(script, "Text Models Evaluation")
        results.append(("Text Evaluation", success))
    else:
        print(f"✗ Script not found: {script}")
        results.append(("Text Evaluation", False))

    # Step 2: Evaluate facial models
    script = scripts_dir / 'evaluate_facial_models.py'
    if script.exists():
        success = run_script(script, "Facial Models Evaluation")
        results.append(("Facial Evaluation", success))
    else:
        print(f"✗ Script not found: {script}")
        results.append(("Facial Evaluation", False))

    # Step 3: Generate report graphs
    script = scripts_dir / 'generate_report_graphs.py'
    if script.exists():
        success = run_script(script, "Report Graphs Generation")
        results.append(("Graph Generation", success))
    else:
        print(f"✗ Script not found: {script}")
        results.append(("Graph Generation", False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for task, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{task:<30} {status}")

    all_success = all(success for _, success in results)

    if all_success:
        print("\n" + "="*70)
        print("ALL TASKS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nGenerated outputs:")
        print("  - evaluation_results/text/     (Text model metrics)")
        print("  - evaluation_results/facial/   (Facial model metrics)")
        print("  - report_graphs/               (All visualizations)")
        print("\nYou can now:")
        print("  1. View confusion matrices in evaluation_results/")
        print("  2. Check accuracy comparisons in report_graphs/")
        print("  3. Use graphs in your 20-page report")
        print("  4. Run the frontend to test all models")
    else:
        print("\n" + "="*70)
        print("SOME TASKS FAILED")
        print("="*70)
        print("\nPlease check the error messages above and:")
        print("  1. Ensure all models are trained")
        print("  2. Verify test data exists in data/ folder")
        print("  3. Check for missing dependencies")

if __name__ == "__main__":
    main()
