"""
JSONL ডেটাসেট ভ্যালিডেশন ও ফিক্স করার স্ক্রিপ্ট
"""

import json
import sys
from pathlib import Path


def validate_and_fix_jsonl(input_file, output_file=None):
    """
    JSONL ফাইল চেক করে এবং সমস্যা ঠিক করে
    
    Args:
        input_file: ইনপুট JSONL ফাইল
        output_file: আউটপুট ফাইল (None হলে _fixed.jsonl যোগ হবে)
    """
    if output_file is None:
        output_file = str(input_file).replace('.jsonl', '_fixed.jsonl')
    
    print(f"📂 ফাইল পড়া হচ্ছে: {input_file}")
    print(f"{'='*60}\n")
    
    valid_data = []
    errors = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # খালি লাইন স্কিপ
            if not line:
                continue
            
            try:
                # JSON parse
                item = json.loads(line)
                
                # ফিল্ড চেক
                if not isinstance(item, dict):
                    errors.append(f"লাইন {line_num}: ডেটা dictionary নয়")
                    continue
                
                if 'instruction' not in item:
                    errors.append(f"লাইন {line_num}: 'instruction' ফিল্ড নেই")
                    continue
                
                if 'output' not in item:
                    errors.append(f"লাইন {line_num}: 'output' ফিল্ড নেই")
                    continue
                
                # খালি ফিল্ড চেক
                if not item['instruction'].strip():
                    errors.append(f"লাইন {line_num}: খালি instruction")
                    continue
                
                if not item['output'].strip():
                    errors.append(f"লাইন {line_num}: খালি output")
                    continue
                
                # ভ্যালিড ডেটা
                valid_data.append({
                    'instruction': item['instruction'].strip(),
                    'output': item['output'].strip()
                })
                
            except json.JSONDecodeError as e:
                errors.append(f"লাইন {line_num}: JSON পার্স এরর - {str(e)[:50]}")
                continue
            except Exception as e:
                errors.append(f"লাইন {line_num}: অজানা এরর - {str(e)[:50]}")
                continue
    
    # রেজাল্ট
    print(f"✅ ভ্যালিড ডেটা: {len(valid_data)} টি")
    print(f"❌ সমস্যাযুক্ত লাইন: {len(errors)} টি\n")
    
    if errors:
        print("⚠️  সমস্যার তালিকা:")
        for i, error in enumerate(errors[:20], 1):  # প্রথম ২০টি দেখান
            print(f"  {i}. {error}")
        if len(errors) > 20:
            print(f"  ... আরও {len(errors) - 20} টি সমস্যা")
        print()
    
    # সঠিক ডেটা সেভ করুন
    if valid_data:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in valid_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"💾 ঠিক করা ফাইল সেভ হয়েছে: {output_file}")
        print(f"📊 মোট লাইন: {len(valid_data)}")
        
        # স্ট্যাটিস্টিক্স
        avg_inst_len = sum(len(d['instruction']) for d in valid_data) / len(valid_data)
        avg_out_len = sum(len(d['output']) for d in valid_data) / len(valid_data)
        
        print(f"\n📈 পরিসংখ্যান:")
        print(f"  - গড় instruction দৈর্ঘ্য: {avg_inst_len:.0f} অক্ষর")
        print(f"  - গড় output দৈর্ঘ্য: {avg_out_len:.0f} অক্ষর")
        
        # উদাহরণ দেখান
        print(f"\n📝 প্রথম ৩টি উদাহরণ:")
        for i, example in enumerate(valid_data[:3], 1):
            print(f"\n{i}. প্রশ্ন: {example['instruction'][:80]}...")
            print(f"   উত্তর: {example['output'][:80]}...")
    else:
        print("❌ কোনো ভ্যালিড ডেটা পাওয়া যায়নি!")
        return False
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="JSONL ডেটাসেট ভ্যালিডেট ও ফিক্স করুন")
    parser.add_argument("input_file", help="ইনপুট JSONL ফাইল")
    parser.add_argument("-o", "--output", help="আউটপুট ফাইল (ডিফল্ট: input_fixed.jsonl)")
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"❌ ফাইল পাওয়া যায়নি: {args.input_file}")
        sys.exit(1)
    
    success = validate_and_fix_jsonl(args.input_file, args.output)
    
    if success:
        print("\n✅ সফল!")
        sys.exit(0)
    else:
        print("\n❌ ব্যর্থ!")
        sys.exit(1)


if __name__ == "__main__":
    main()
