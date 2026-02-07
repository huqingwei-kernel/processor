#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理器五级流水线模拟器 - 无乱码版本
使用ASCII字符避免字体问题
"""

from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass
import time
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False
    print("提示: 未安装tkinter，GUI功能不可用")
    print("使用命令安装: sudo apt-get install python3-tk (Ubuntu/Debian)")
import threading
import queue


class Opcode(Enum):
    ADD = "ADD"
    SUB = "SUB"
    LD = "LD"
    ST = "ST"
    BEQ = "BEQ"
    NOP = "NOP"


@dataclass
class Instruction:
    opcode: Opcode
    rd: Optional[int] = None
    rs1: Optional[int] = None
    rs2: Optional[int] = None
    imm: Optional[int] = None
    label: Optional[str] = None

    def __str__(self):
        if self.opcode in [Opcode.ADD, Opcode.SUB]:
            return f"{self.opcode.value} R{self.rd}, R{self.rs1}, R{self.rs2}"
        elif self.opcode == Opcode.LD:
            return f"{self.opcode.value} R{self.rd}, {self.imm}(R{self.rs1})"
        elif self.opcode == Opcode.ST:
            return f"{self.opcode.value} R{self.rs1}, {self.imm}(R{self.rs2})"
        elif self.opcode == Opcode.BEQ:
            return f"{self.opcode.value} R{self.rs1}, R{self.rs2}, {self.label}"
        else:
            return self.opcode.value


@dataclass
class PipelineRegister:
    instruction: Optional[Instruction] = None
    pc: int = 0
    alu_result: int = 0
    read_data1: int = 0
    read_data2: int = 0
    stalled: bool = False
    bubble: bool = False
    # For ALU stage result
    alu_output: int = 0


class PipelineSimulator:
    def __init__(self):
        self.registers = [0] * 32
        self.memory = [0] * 256
        self.pc = 0
        self.if_id = PipelineRegister()
        self.id_ex = PipelineRegister()
        self.rd_alu = PipelineRegister()  # RD stage: Register Read
        self.alu_mem = PipelineRegister() # ALU stage: ALU operations
        self.mem_wb = PipelineRegister()
        self.cycle_count = 0
        self.instruction_count = 0
        self.stall_count = 0
        self.bubble_count = 0
        self.instructions: List[Instruction] = []
        self.labels: Dict[str, int] = {}
        self.data_hazard = False
        self.control_hazard = False

    def load_program(self, program: List[Instruction]):
        self.instructions = program.copy()
        self.labels["LOOP"] = 0
        self.labels["END"] = len(program) - 1

    def fetch(self):
        if self.pc >= len(self.instructions):
            return Instruction(Opcode.NOP), True
        instruction = self.instructions[self.pc]
        return instruction, False

    def decode(self, instruction: Instruction):
        read_data1 = self.registers[instruction.rs1] if instruction.rs1 is not None else 0
        read_data2 = self.registers[instruction.rs2] if instruction.rs2 is not None else 0
        return read_data1, read_data2

    def execute(self, instruction: Instruction, read_data1: int, read_data2: int):
        if instruction.opcode == Opcode.ADD:
            return read_data1 + read_data2
        elif instruction.opcode == Opcode.SUB:
            return read_data1 - read_data2
        elif instruction.opcode == Opcode.LD:
            return read_data1 + (instruction.imm or 0)
        elif instruction.opcode == Opcode.ST:
            return read_data2 + (instruction.imm or 0)
        elif instruction.opcode == Opcode.BEQ:
            return 1 if read_data1 == read_data2 else 0
        return 0

    def memory_access(self, instruction: Instruction, alu_result: int, read_data2: int):
        if instruction.opcode == Opcode.LD:
            return self.memory[alu_result % len(self.memory)]
        elif instruction.opcode == Opcode.ST:
            self.memory[alu_result % len(self.memory)] = read_data2
        return alu_result

    def write_back(self, instruction: Instruction, write_data: int):
        if instruction.opcode in [Opcode.ADD, Opcode.SUB, Opcode.LD]:
            if instruction.rd is not None and instruction.rd != 0:
                self.registers[instruction.rd] = write_data

    def detect_hazards(self):
        self.data_hazard = False
        self.control_hazard = False

        # Check for data hazard: RD stage reads registers that ALU stage will write to
        if (self.rd_alu.instruction and self.rd_alu.instruction.rd is not None and
            self.id_ex.instruction):
            if (self.rd_alu.instruction.rd == self.id_ex.instruction.rs1 or
                self.rd_alu.instruction.rd == self.id_ex.instruction.rs2):
                self.data_hazard = True

        # Check for control hazard: ALU stage has branch instruction
        if (self.rd_alu.instruction and
            self.rd_alu.instruction.opcode in [Opcode.BEQ]):
            self.control_hazard = True

    def step(self):
        self.cycle_count += 1

        # WB
        if self.mem_wb.instruction and not self.mem_wb.bubble:
            self.write_back(self.mem_wb.instruction, self.mem_wb.alu_result)
            if self.mem_wb.instruction.opcode != Opcode.NOP:
                self.instruction_count += 1

        # MEM
        if self.alu_mem.instruction:
            if not self.alu_mem.bubble:
                write_data = self.memory_access(
                    self.alu_mem.instruction,
                    self.alu_mem.alu_output,  # Use ALU output
                    self.alu_mem.read_data2
                )
                self.mem_wb = PipelineRegister(
                    instruction=self.alu_mem.instruction,
                    alu_result=write_data,
                    bubble=self.alu_mem.bubble
                )
            else:
                self.mem_wb = PipelineRegister(bubble=True)
        else:
            self.mem_wb = PipelineRegister()

        # ALU
        if self.rd_alu.instruction and not self.rd_alu.stalled:
            if not self.rd_alu.bubble:
                alu_result = self.execute(
                    self.rd_alu.instruction,
                    self.rd_alu.read_data1,
                    self.rd_alu.read_data2
                )
                self.alu_mem = PipelineRegister(
                    instruction=self.rd_alu.instruction,
                    alu_output=alu_result,  # Store ALU result
                    read_data2=self.rd_alu.read_data2,
                    bubble=self.rd_alu.bubble
                )

                # branch
                if (self.rd_alu.instruction.opcode == Opcode.BEQ and
                    alu_result == 1 and
                    self.rd_alu.instruction.label):
                    self.pc = self.labels.get(self.rd_alu.instruction.label, self.pc + 1)
                    self.rd_alu.bubble = True
                    self.if_id.bubble = True
            else:
                self.alu_mem = PipelineRegister(bubble=True)
        elif self.rd_alu.stalled:
            self.alu_mem = PipelineRegister(bubble=True)
            self.stall_count += 1
        else:
            self.alu_mem = PipelineRegister()

        # RD
        if self.id_ex.instruction and not self.id_ex.stalled:
            if not self.id_ex.bubble:
                # Read register values
                read_data1, read_data2 = self.decode(self.id_ex.instruction)
                self.rd_alu = PipelineRegister(
                    instruction=self.id_ex.instruction,
                    pc=self.id_ex.pc,
                    read_data1=read_data1,
                    read_data2=read_data2,
                    bubble=self.id_ex.bubble
                )
            else:
                self.rd_alu = PipelineRegister(bubble=True)
        else:
            self.rd_alu = PipelineRegister()

        # ID
        if self.if_id.instruction and not self.if_id.stalled:
            if self.if_id.bubble:
                self.id_ex = PipelineRegister(bubble=True)
                self.bubble_count += 1
            else:
                # Only decode instruction in ID stage
                self.id_ex = PipelineRegister(
                    instruction=self.if_id.instruction,
                    pc=self.if_id.pc,
                    bubble=self.if_id.bubble
                )
        else:
            self.id_ex = PipelineRegister()

        # IF
        if not self.rd_alu.stalled:
            instruction, finished = self.fetch()
            if not finished:
                self.if_id = PipelineRegister(instruction=instruction, pc=self.pc)
                self.pc += 1
            else:
                self.if_id = PipelineRegister(instruction=instruction)
        else:
            self.if_id = PipelineRegister()

        # Detect and handle hazards
        self.detect_hazards()

        if self.data_hazard:
            self.if_id.stalled = True
            self.id_ex.stalled = False
        elif self.control_hazard:
            self.if_id = PipelineRegister(bubble=True)
        else:
            self.if_id.stalled = False
            self.id_ex.stalled = False

    def print_pipeline_diagram(self):
        """Print pipeline timing diagram"""
        print(f"\n{'='*80}")
        print(f"Cycle: {self.cycle_count}")
        print(f"{'='*80}")

        # Show pipeline register status
        stages = {
            "IF": self.if_id,
            "ID": self.id_ex,
            "RD": self.rd_alu,
            "ALU": self.alu_mem,
            "MEM": self.mem_wb,
            "WB": "Done"
        }

        for stage_name, stage_data in stages.items():
            if stage_name == "WB":
                print(f"{stage_name}: {'Done':<30}")
            elif stage_data.bubble:
                print(f"{stage_name}: {'Bubble':<30}")
            elif stage_data.instruction:
                print(f"{stage_name}: {str(stage_data.instruction):<30}", end="")
                if stage_name == "RD" and self.data_hazard:
                    print(" [Data Hazard]", end="")
                elif stage_name == "ALU" and self.control_hazard:
                    print(" [Control Hazard]", end="")
                print()
            else:
                print(f"{stage_name}: {'Empty':<30}")

        # Show registers
        print("\nRegisters (important):")
        important_regs = []
        for i in range(8):
            if self.registers[i] != 0:
                important_regs.append(f"R{i}={self.registers[i]}")
        if important_regs:
            print(", ".join(important_regs))
        else:
            print("No changes")

        # Show statistics
        print(f"\nStats: Instructions={self.instruction_count}, Stalls={self.stall_count}, Bubbles={self.bubble_count}")

    def run(self, max_cycles=15, delay=0.3):
        """Run simulation"""
        print("\n" + "="*70)
        print("5-Stage Pipeline Simulation")
        print("="*70)

        for cycle in range(max_cycles):
            self.step()
            self.print_pipeline_diagram()

            if delay > 0:
                time.sleep(delay)

            # Check if done
            real_instructions = [i for i in self.instructions if i.opcode != Opcode.NOP]
            if (self.instruction_count >= len(real_instructions) and
                not any([self.if_id.instruction, self.id_ex.instruction,
                        self.rd_alu.instruction, self.alu_mem.instruction, self.mem_wb.instruction])):
                print("\nAll instructions completed!")
                break

        # Final results
        print("\n" + "="*70)
        print("Simulation Complete!")
        print(f"Total cycles: {self.cycle_count}")
        print(f"Total instructions: {self.instruction_count}")
        print(f"Stall count: {self.stall_count}")
        print(f"Bubble count: {self.bubble_count}")
        cpi = self.cycle_count / max(self.instruction_count, 1)
        print(f"CPI: {cpi:.2f}")

        print("\nFinal register values:")
        for i in range(8):
            print(f"R{i}: {self.registers[i]:6d}", end="    ")
            if i % 4 == 3:
                print()


class PipelineSimulatorGUI:
    """Pipeline simulator with GUI"""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Pipeline Simulator")
        self.window.geometry("1200x800")

        # Color scheme
        self.colors = {
            'IF': '#E8F5E9',
            'ID': '#FFF3E0',
            'RD': '#E1F5FE',
            'ALU': '#E3F2FD',
            'MEM': '#F3E5F5',
            'WB': '#FFEBEE',
            'bubble': '#FFCDD2',
            'stall': '#FFECB3',
            'hazard': '#FF8A80'
        }

        self.simulator = PipelineSimulator()
        self.is_running = False
        self.speed = 1.0
        self.auto_step_id = None

        self.setup_ui()
        self.load_default_program()

    def setup_ui(self):
        """Setup user interface"""
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel (control)
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Control buttons
        ttk.Label(control_frame, text="Control Panel", font=('TkDefaultFont', 12, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start_simulation)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.pause_btn = ttk.Button(btn_frame, text="Pause", command=self.pause_simulation, state='disabled')
        self.pause_btn.grid(row=0, column=1, padx=5)

        self.step_btn = ttk.Button(btn_frame, text="Step", command=self.step_once)
        self.step_btn.grid(row=0, column=2, padx=5)

        self.reset_btn = ttk.Button(btn_frame, text="Reset", command=self.reset_simulation)
        self.reset_btn.grid(row=0, column=3, padx=5)

        # Speed control
        speed_frame = ttk.LabelFrame(control_frame, text="Speed Control", padding="10")
        speed_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.speed_scale = ttk.Scale(speed_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                                     value=1.0, command=self.on_speed_change)
        self.speed_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.grid(row=0, column=1, padx=10)

        # Current instruction
        inst_frame = ttk.LabelFrame(control_frame, text="Current Instruction", padding="10")
        inst_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.current_inst_label = ttk.Label(inst_frame, text="Waiting...",
                                          wraplength=200, justify=tk.LEFT)
        self.current_inst_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        stats = [
            ("Clock Cycles:", "cycle_count"),
            ("Instructions:", "instruction_count"),
            ("Stalls:", "stall_count"),
            ("Bubbles:", "bubble_count"),
            ("CPI:", "cpi")
        ]

        self.stats_labels = {}
        row = 0
        for label, key in stats:
            ttk.Label(stats_frame, text=label).grid(row=row, column=0, sticky=tk.W)
            self.stats_labels[key] = ttk.Label(stats_frame, text="0", width=10)
            self.stats_labels[key].grid(row=row, column=1, sticky=tk.W)
            row += 1

        # Center panel (pipeline visualization)
        pipeline_frame = ttk.LabelFrame(main_frame, text="Pipeline Visualization", padding="10")
        pipeline_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Pipeline stages display
        self.stage_labels = {}
        self.stage_frames = {}

        stages = [("IF (Fetch)", "IF"), ("ID (Decode)", "ID"), ("RD (Read Reg)", "RD"),
                 ("ALU (Execute)", "ALU"), ("MEM (Memory)", "MEM"), ("WB (WriteBack)", "WB")]

        for i, (stage_display, stage_key) in enumerate(stages):
            # Stage label
            ttk.Label(pipeline_frame, text=stage_display, font=('TkDefaultFont', 10, 'bold')).grid(
                row=0, column=i, padx=5, pady=5)

            # Stage content frame
            frame = ttk.Frame(pipeline_frame, width=180, height=60, relief='solid', borderwidth=1)
            frame.grid(row=1, column=i, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
            frame.grid_propagate(False)

            # Stage content label
            label = ttk.Label(frame, text="Empty", wraplength=150, justify=tk.CENTER)
            label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

            self.stage_labels[stage_key] = label
            self.stage_frames[stage_key] = frame

        # Hazard warning label
        self.hazard_label = ttk.Label(pipeline_frame, text="", foreground='red', font=('TkDefaultFont', 10, 'bold'))
        self.hazard_label.grid(row=2, column=0, columnspan=5, pady=10)

        # Right register display
        reg_frame = ttk.LabelFrame(main_frame, text="Register Status", padding="10")
        reg_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Create register labels
        self.reg_labels = []
        for i in range(8):
            row = i // 2
            col = (i % 2) * 2

            ttk.Label(reg_frame, text=f"R{i}:", font=('TkDefaultFont', 9)).grid(
                row=row, column=col, sticky=tk.W, padx=2)

            reg_label = ttk.Label(reg_frame, text="00000", width=6, font=('Courier', 9))
            reg_label.grid(row=row, column=col+1, sticky=tk.W, padx=2)
            self.reg_labels.append(reg_label)

        # Bottom (program list)
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        ttk.Label(bottom_frame, text="Instruction List", font=('TkDefaultFont', 11, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=5)

        # Instruction list text box
        self.inst_text = scrolledtext.ScrolledText(bottom_frame, height=8, width=140, font=('Courier', 9))
        self.inst_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure column weights
        for i in range(3):
            main_frame.columnconfigure(i, weight=1)
        main_frame.rowconfigure(0, weight=1)
        bottom_frame.columnconfigure(0, weight=1)

    def on_speed_change(self, value):
        """Speed slider change event"""
        self.speed = float(value)
        self.speed_label.config(text=f"{self.speed:.1f}x")

    def load_default_program(self):
        """Load default program"""
        program = [
            Instruction(Opcode.LD, rd=1, rs1=0, imm=100),
            Instruction(Opcode.ADD, rd=2, rs1=1, rs2=0),
            Instruction(Opcode.ADD, rd=3, rs1=2, rs2=1),
            Instruction(Opcode.SUB, rd=4, rs1=1, rs2=2),
            Instruction(Opcode.ST, rs1=4, rs2=0, imm=200),
            Instruction(Opcode.BEQ, rs1=3, rs2=4, label="LOOP"),
            Instruction(Opcode.ADD, rd=5, rs1=1, rs2=2),
            Instruction(Opcode.NOP),
        ]

        self.simulator.memory[100] = 10
        self.simulator.load_program(program)

        # Display instructions in text box
        self.inst_text.delete(1.0, tk.END)
        for i, instr in enumerate(program):
            self.inst_text.insert(tk.END, f"{i:2d}: {instr}\n")
        self.inst_text.config(state='disabled')

    def _update_frame_bg(self, frame, color):
        """Update frame background color"""
        style_name = f'ColorFrame_{color.replace("#", "")}.TFrame'
        if not hasattr(frame, '_style') or frame._style != style_name:
            style = ttk.Style()
            style.configure(style_name, background=color)
            frame.config(style=style_name)
            frame._style = style_name

            # Update background of all children
            for child in frame.winfo_children():
                if isinstance(child, ttk.Label):
                    child.config(background=color)

    def update_display(self):
        """Update display"""
        self._update_display_internal()

    def _update_display_internal(self):
        """Internal display update"""
        # Update pipeline stage display
        stages = ["IF", "ID", "RD", "ALU", "MEM", "WB"]
        pipeline_regs = [self.simulator.if_id, self.simulator.id_ex,
                        self.simulator.rd_alu, self.simulator.alu_mem, self.simulator.mem_wb]

        for i, stage in enumerate(stages):
            if stage == "WB":
                # Write-back stage
                if self.simulator.mem_wb.instruction and not self.simulator.mem_wb.bubble:
                    text = "DONE\n" + str(self.simulator.mem_wb.instruction)[:25]
                    bg_color = self.colors['WB']
                else:
                    text = "Waiting"
                    bg_color = 'white'
            else:
                # Other stages
                reg = pipeline_regs[i]
                if reg.bubble:
                    text = "BUBBLE"
                    bg_color = self.colors['bubble']
                elif reg.instruction:
                    text = str(reg.instruction)
                    if stage == "RD" and self.simulator.data_hazard:
                        bg_color = self.colors['hazard']
                    elif stage == "ALU" and self.simulator.control_hazard:
                        bg_color = self.colors['hazard']
                    else:
                        bg_color = self.colors.get(stage, 'white')
                else:
                    text = "Empty"
                    bg_color = 'white'

                if reg and hasattr(reg, 'stalled') and reg.stalled:
                    bg_color = self.colors['stall']

            self.stage_labels[stage].config(text=text, font=('TkDefaultFont', 9))
            self.stage_frames[stage].config(style=f'{stage}.TFrame')
            # Update background color
            self._update_frame_bg(self.stage_frames[stage], bg_color)

        # Update hazard warning
        if self.simulator.data_hazard:
            self.hazard_label.config(text="WARNING: Data Hazard Detected!")
        elif self.simulator.control_hazard:
            self.hazard_label.config(text="WARNING: Control Hazard Detected!")
        else:
            self.hazard_label.config(text="")

        # Update current instruction
        if self.simulator.if_id.instruction:
            self.current_inst_label.config(text=f"PC={self.simulator.pc}: {self.simulator.if_id.instruction}")
        elif self.simulator.id_ex.instruction:
            self.current_inst_label.config(text=f"Decode: {self.simulator.id_ex.instruction}")
        elif self.simulator.alu_mem.instruction:
            self.current_inst_label.config(text=f"ALU: {self.simulator.alu_mem.instruction}")
        elif self.simulator.mem_wb.instruction:
            self.current_inst_label.config(text=f"Memory: {self.simulator.mem_wb.instruction}")

        # Update statistics
        self.stats_labels['cycle_count'].config(text=str(self.simulator.cycle_count))
        self.stats_labels['instruction_count'].config(text=str(self.simulator.instruction_count))
        self.stats_labels['stall_count'].config(text=str(self.simulator.stall_count))
        self.stats_labels['bubble_count'].config(text=str(self.simulator.bubble_count))

        cpi = self.simulator.cycle_count / max(self.simulator.instruction_count, 1)
        self.stats_labels['cpi'].config(text=f"{cpi:.2f}")

        # Update registers
        for i in range(8):
            self.reg_labels[i].config(text=f"{self.simulator.registers[i]:5d}")

        self.window.update()

    def start_simulation(self):
        """Start simulation"""
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(state='disabled')
            self.pause_btn.config(state='normal')
            self.step_btn.config(state='disabled')
            self.auto_step()

    def pause_simulation(self):
        """Pause simulation"""
        self.is_running = False
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.step_btn.config(state='normal')

        if self.auto_step_id:
            self.window.after_cancel(self.auto_step_id)
            self.auto_step_id = None

    def step_once(self):
        """Single step"""
        self.simulator.step()
        self._update_display_internal()

    def auto_step(self):
        """Auto step"""
        if self.is_running:
            # Check if complete
            real_instructions = [i for i in self.simulator.instructions if i.opcode != Opcode.NOP]
            if (self.simulator.instruction_count >= len(real_instructions) and
                not any([self.simulator.if_id.instruction, self.simulator.id_ex.instruction,
                        self.simulator.rd_alu.instruction, self.simulator.alu_mem.instruction, self.simulator.mem_wb.instruction])):
                self.pause_simulation()
                messagebox.showinfo("Complete", "Simulation Complete!")
                return

            self.simulator.step()
            self._update_display_internal()

            # Calculate delay
            delay = int(500 / self.speed)
            self.auto_step_id = self.window.after(delay, self.auto_step)

    def reset_simulation(self):
        """Reset simulation"""
        self.pause_simulation()
        self.simulator = PipelineSimulator()
        self.load_default_program()
        self._update_display_internal()

    def run(self):
        """Run GUI"""
        if not TK_AVAILABLE:
            print("Error: tkinter not available")
            return

        self._update_display_internal()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def on_closing(self):
        """Window close event"""
        self.is_running = False
        if self.auto_step_id:
            self.window.after_cancel(self.auto_step_id)
        self.window.destroy()


def demo():
    """Demo program - command line version"""
    # Create program with data hazards
    program = [
        Instruction(Opcode.LD, rd=1, rs1=0, imm=100),   # Cycle 1: Load R1
        Instruction(Opcode.ADD, rd=2, rs1=1, rs2=0),    # Cycle 2: R2 = R1 + 0 (Data Hazard!)
        Instruction(Opcode.ADD, rd=3, rs1=2, rs2=1),    # Cycle 3: R3 = R2 + R1 (Data Hazard!)
        Instruction(Opcode.SUB, rd=4, rs1=1, rs2=2),    # R4 = R1 - R2
        Instruction(Opcode.ST, rs1=4, rs2=0, imm=200),  # Store R4
        Instruction(Opcode.BEQ, rs1=3, rs2=4, label="LOOP"),  # Branch (Control Hazard!)
        Instruction(Opcode.ADD, rd=5, rs1=1, rs2=2),    # R5 = R1 + R2
        Instruction(Opcode.NOP),
    ]

    print("\n5-Stage Pipeline Simulation Demo")
    print("="*70)
    print("\nProgram Instructions:")
    for i, instr in enumerate(program):
        print(f"{i:2d}: {instr}")

    print("\nInstruction types:")
    print("- LD: Load")
    print("- ADD/SUB: Arithmetic")
    print("- ST: Store")
    print("- BEQ: Branch if Equal")
    print("- Demonstrates data and control hazards")

    # Create simulator
    sim = PipelineSimulator()

    # Initialize memory
    sim.memory[100] = 10

    # Load program
    sim.load_program(program)

    # Run
    input("\nPress Enter to start simulation...")
    sim.run(max_cycles=20, delay=0.5)

    print("\nSimulation Complete!")


def gui_demo():
    """GUI demo"""
    gui = PipelineSimulatorGUI()
    gui.run()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        gui_demo()
    else:
        print("Select mode:")
        print("1. Command Line")
        print("2. GUI")
        choice = input("Enter choice (1-2): ").strip()
        if choice == '2':
            gui_demo()
        else:
            demo()
