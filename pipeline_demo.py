#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理器六级流水线模拟器 - 最终版本
支持图形化界面，实时显示寄存器和内存状态
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
            # For ST: rs1 is the register to store, rs2 is the base address register
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

        # Pipeline registers (buffers between stages)
        self.if_id = PipelineRegister()   # Buffer between IF and ID
        self.id_rd = PipelineRegister()   # Buffer between ID and RD
        self.rd_alu = PipelineRegister()  # Buffer between RD and ALU
        self.alu_mem = PipelineRegister() # Buffer between ALU and MEM
        self.mem_wb = PipelineRegister()  # Buffer between MEM and WB

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

    def memory_access(self, instruction: Instruction, alu_result: int, read_data1: int, read_data2: int):
        if instruction.opcode == Opcode.LD:
            value = self.memory[alu_result % len(self.memory)]
            return value
        elif instruction.opcode == Opcode.ST:
            # For ST: read_data1 is the value to store, read_data2 is base address register value
            addr = alu_result % len(self.memory)
            self.memory[addr] = read_data1
        return alu_result

    def write_back(self, instruction: Instruction, write_data: int):
        if instruction.opcode in [Opcode.ADD, Opcode.SUB, Opcode.LD]:
            if instruction.rd is not None and instruction.rd != 0:
                self.registers[instruction.rd] = write_data

    def detect_hazards(self):
        # Simplified: no hazard detection for now
        # In a real pipeline, we would detect data and control hazards here
        self.data_hazard = False
        self.control_hazard = False

    def step(self):
        self.cycle_count += 1

        # === Stage 6: WB (Write Back) ===
        # WB stage writes to register file
        if self.mem_wb.instruction and not self.mem_wb.bubble:
            self.write_back(self.mem_wb.instruction, self.mem_wb.alu_result)
            if self.mem_wb.instruction.opcode != Opcode.NOP:
                self.instruction_count += 1

        # === Stage 5: MEM (Memory Access) ===
        # MEM reads from ALU/MEM buffer, writes to MEM/WB buffer
        if self.alu_mem.instruction:
            if not self.alu_mem.bubble:
                # Memory access for LD/ST instructions
                write_data = self.memory_access(
                    self.alu_mem.instruction,
                    self.alu_mem.alu_output,  # Address for LD/ST
                    self.alu_mem.read_data1,  # Value to store (for ST)
                    self.alu_mem.read_data2   # Base register value
                )

                # Write to MEM/WB buffer
                self.mem_wb = PipelineRegister(
                    instruction=self.alu_mem.instruction,
                    alu_result=write_data if self.alu_mem.instruction.opcode != Opcode.ST else 0,
                    bubble=self.alu_mem.bubble
                )
            else:
                self.mem_wb = PipelineRegister(bubble=True)
        else:
            self.mem_wb = PipelineRegister()

        # === Stage 4: ALU (Execute) ===
        # ALU reads from RD/ALU buffer, writes to ALU/MEM buffer
        if self.rd_alu.instruction and not self.rd_alu.stalled:
            if not self.rd_alu.bubble:
                # Execute ALU operation
                alu_result = self.execute(
                    self.rd_alu.instruction,
                    self.rd_alu.read_data1,
                    self.rd_alu.read_data2
                )

                # Write to ALU/MEM buffer
                self.alu_mem = PipelineRegister(
                    instruction=self.rd_alu.instruction,
                    alu_output=alu_result,        # ALU result or memory address
                    read_data1=self.rd_alu.read_data1,  # Value for ST instruction
                    read_data2=self.rd_alu.read_data2,  # Base register
                    bubble=self.rd_alu.bubble
                )

                # Handle branch in ALU stage
                if (self.rd_alu.instruction.opcode == Opcode.BEQ and
                    alu_result == 1 and
                    self.rd_alu.instruction.label):
                    self.pc = self.labels.get(self.rd_alu.instruction.label, self.pc + 1)
                    # Insert bubbles in earlier stages
                    self.rd_alu.bubble = True
                    self.id_rd.bubble = True
                    self.if_id.bubble = True
            else:
                self.alu_mem = PipelineRegister(bubble=True)
        elif self.rd_alu.stalled:
            # Stall: insert bubble and increment stall count
            self.alu_mem = PipelineRegister(bubble=True)
            self.stall_count += 1
        else:
            self.alu_mem = PipelineRegister()

        # === Stage 3: RD (Register Read) ===
        # RD reads from ID/RD buffer, writes to RD/ALU buffer
        if self.id_rd.instruction and not self.id_rd.stalled:
            if not self.id_rd.bubble:
                # Read register values
                read_data1, read_data2 = self.decode(self.id_rd.instruction)

                # Write to RD/ALU buffer
                self.rd_alu = PipelineRegister(
                    instruction=self.id_rd.instruction,
                    pc=self.id_rd.pc,
                    read_data1=read_data1,
                    read_data2=read_data2,
                    bubble=self.id_rd.bubble
                )
            else:
                self.rd_alu = PipelineRegister(bubble=True)
        else:
            self.rd_alu = PipelineRegister()

        # === Stage 2: ID (Instruction Decode) ===
        # ID reads from IF/ID buffer, writes to ID/RD buffer
        if self.if_id.instruction and not self.if_id.stalled:
            if self.if_id.bubble:
                # Bubble propagates to next buffer
                self.id_rd = PipelineRegister(bubble=True)
                self.bubble_count += 1
            else:
                # Decode instruction and write to ID/RD buffer
                self.id_rd = PipelineRegister(
                    instruction=self.if_id.instruction,
                    pc=self.if_id.pc,
                    bubble=self.if_id.bubble
                )
        else:
            self.id_rd = PipelineRegister()

        # === Stage 1: IF (Instruction Fetch) ===
        # IF fetches instruction and writes to IF/ID buffer
        if not self.if_id.stalled:
            instruction, finished = self.fetch()
            if not finished:
                self.if_id = PipelineRegister(instruction=instruction, pc=self.pc)
                self.pc += 1
            else:
                self.if_id = PipelineRegister(instruction=instruction)
        else:
            self.if_id = PipelineRegister()

        # === Hazard Detection and Handling ===
        self.detect_hazards()

        if self.data_hazard:
            # Data hazard: stall IF and ID stages
            self.if_id.stalled = True
            self.id_rd.stalled = False  # Allow bubble to propagate
        elif self.control_hazard:
            # Control hazard: insert bubble in IF
            self.if_id = PipelineRegister(bubble=True)
        else:
            # No hazards: all stages unstalled
            self.if_id.stalled = False
            self.id_rd.stalled = False

    def print_pipeline_diagram(self):
        """Print pipeline timing diagram with buffers"""
        print(f"\n{'='*80}")
        print(f"Cycle: {self.cycle_count}")
        print(f"{'='*80}")

        # Show buffers between stages
        print("Pipeline Buffers:")
        print(f"  IF/ID: {str(self.if_id.instruction) if self.if_id.instruction else 'Empty':<30} {'[Stalled]' if self.if_id.stalled else ''}")
        print(f"  ID/RD: {str(self.id_rd.instruction) if self.id_rd.instruction else 'Empty':<30} {'[Stalled]' if self.id_rd.stalled else ''}")
        print(f"  RD/ALU: {str(self.rd_alu.instruction) if self.rd_alu.instruction else 'Empty':<30} {'[Stalled]' if self.rd_alu.stalled else ''}")
        print(f"  ALU/MEM: {str(self.alu_mem.instruction) if self.alu_mem.instruction else 'Empty':<30}")
        print(f"  MEM/WB: {str(self.mem_wb.instruction) if self.mem_wb.instruction else 'Empty':<30}")

        # Show pipeline stages (what's currently in each stage)
        print("\nPipeline Stages:")
        stages = {
            "IF": self.if_id,
            "ID": self.id_rd,
            "RD": self.rd_alu,
            "ALU": self.alu_mem,
            "MEM": self.mem_wb,
            "WB": "Done"
        }

        for stage_name, stage_data in stages.items():
            if stage_name == "WB":
                print(f"{stage_name}: {'Done':<30}")
            elif hasattr(stage_data, 'bubble') and stage_data.bubble:
                print(f"{stage_name}: {'Bubble':<30}")
            elif hasattr(stage_data, 'instruction') and stage_data.instruction:
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
                not any([self.if_id.instruction, self.id_rd.instruction,
                        self.rd_alu.instruction, self.mem_wb.instruction])):
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
        """Setup user interface with vertical pipeline and resizable panes"""
        # Use PanedWindow for resizable sections
        main_paned = tk.PanedWindow(self.window, orient=tk.HORIZONTAL,
                                   sashwidth=10, sashrelief=tk.RAISED,
                                   bg='#E0E0E0')
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left section - Control Panel
        left_frame = ttk.LabelFrame(main_paned, text="Control Panel", padding="10")
        main_paned.add(left_frame, minsize=200)

        # Control buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=0, column=0, columnspan=2, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = ttk.Button(btn_frame, text="Pause", command=self.pause_simulation, state='disabled')
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.step_btn = ttk.Button(btn_frame, text="Step", command=self.step_once)
        self.step_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(btn_frame, text="Reset", command=self.reset_simulation)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # Speed control
        speed_frame = ttk.LabelFrame(left_frame, text="Speed Control", padding="10")
        speed_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.speed_scale = ttk.Scale(speed_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                                     value=1.0, command=self.on_speed_change)
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT, padx=5)

        # Current instruction
        inst_frame = ttk.LabelFrame(left_frame, text="Current Instruction", padding="10")
        inst_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.current_inst_label = ttk.Label(inst_frame, text="Waiting...",
                                          wraplength=200, justify=tk.LEFT)
        self.current_inst_label.pack(fill=tk.X)

        # Statistics
        stats_frame = ttk.LabelFrame(left_frame, text="Statistics", padding="10")
        stats_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

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
            self.stats_labels[key].grid(row=row, column=1, sticky=tk.W, padx=5)
            row += 1

        # Middle section - Pipeline Visualization (Vertical)
        middle_frame = ttk.Frame(main_paned)
        main_paned.add(middle_frame, minsize=300)

        # Pipeline Visualization (Vertical layout)
        pipeline_frame = ttk.LabelFrame(middle_frame, text="Pipeline Visualization (Vertical)", padding="10")
        pipeline_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas with scrollbar for pipeline stages
        pipeline_canvas = tk.Canvas(pipeline_frame, width=200)
        pipeline_scrollbar = ttk.Scrollbar(pipeline_frame, orient=tk.VERTICAL, command=pipeline_canvas.yview)
        pipeline_inner_frame = ttk.Frame(pipeline_canvas)

        pipeline_canvas.configure(yscrollcommand=pipeline_scrollbar.set)
        pipeline_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pipeline_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add inner frame to canvas
        pipeline_canvas.create_window((0, 0), window=pipeline_inner_frame, anchor=tk.NW)
        pipeline_inner_frame.bind("<Configure>", lambda e: pipeline_canvas.configure(
            scrollregion=pipeline_canvas.bbox("all")))

        # Pipeline stages display (vertical)
        self.stage_labels = {}
        self.stage_frames = {}

        stages = [("IF (Fetch)", "IF"), ("ID (Decode)", "ID"), ("RD (Read Reg)", "RD"),
                 ("ALU (Execute)", "ALU"), ("MEM (Memory)", "MEM"), ("WB (WriteBack)", "WB")]

        for i, (stage_display, stage_key) in enumerate(stages):
            # Stage label (larger font)
            ttk.Label(pipeline_inner_frame, text=stage_display, font=('TkDefaultFont', 11, 'bold')).grid(
                row=i, column=0, padx=10, pady=5, sticky=tk.W)

            # Stage content frame (larger size to accommodate register values)
            frame = ttk.Frame(pipeline_inner_frame, width=200, height=80, relief='solid', borderwidth=1)
            frame.grid(row=i, column=1, padx=10, pady=5, sticky=(tk.W, tk.E))

            # Stage content label (larger font and wrap length)
            label = ttk.Label(frame, text="Empty", wraplength=180, justify=tk.LEFT, font=('TkDefaultFont', 9))
            label.place(relx=0, rely=0.5, anchor=tk.W)

            self.stage_labels[stage_key] = label
            self.stage_frames[stage_key] = frame

        # Hazard warning label (larger font)
        self.hazard_label = ttk.Label(pipeline_inner_frame, text="", foreground='red',
                                    font=('TkDefaultFont', 10, 'bold'))
        self.hazard_label.grid(row=len(stages), column=0, columnspan=2, pady=10)

        # Right section - Register and Memory Display (in PanedWindow)
        right_paned = tk.PanedWindow(main_paned, orient=tk.VERTICAL,
                                   sashwidth=10, sashrelief=tk.RAISED,
                                   bg='#E0E0E0')
        main_paned.add(right_paned, minsize=400)

        # Register Status
        reg_frame = ttk.LabelFrame(right_paned, text="Register Status", padding="10")
        right_paned.add(reg_frame, minsize=150)

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

        # Memory Status (256 bytes)
        mem_frame = ttk.LabelFrame(right_paned, text="Memory Status (256 bytes)", padding="10")
        right_paned.add(mem_frame, minsize=200)

        # Create canvas with scrollbar for memory display
        mem_canvas = tk.Canvas(mem_frame, width=380, height=250)
        mem_scrollbar = ttk.Scrollbar(mem_frame, orient=tk.VERTICAL, command=mem_canvas.yview)
        self.mem_display_frame = ttk.Frame(mem_canvas)

        mem_canvas.configure(yscrollcommand=mem_scrollbar.set)
        mem_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        mem_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add frame to canvas
        mem_canvas.create_window((0, 0), window=self.mem_display_frame, anchor=tk.NW)
        self.mem_display_frame.bind("<Configure>", lambda e: mem_canvas.configure(
            scrollregion=mem_canvas.bbox("all")))

        # Create memory labels storage - all 256 addresses
        self.mem_labels = {}
        self._create_full_memory_display()

        # Middle-bottom section - Pipeline Buffers
        buffer_frame = ttk.LabelFrame(middle_frame, text="Pipeline Buffers Status", padding="10")
        buffer_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create buffer display labels
        self.buffer_labels = {}
        buffer_names = [("IF/ID", "if_id"), ("ID/RD", "id_rd"), ("RD/ALU", "rd_alu"),
                       ("ALU/MEM", "alu_mem"), ("MEM/WB", "mem_wb")]

        for i, (buffer_name, buffer_key) in enumerate(buffer_names):
            # Buffer label
            ttk.Label(buffer_frame, text=f"{buffer_name}:", font=('TkDefaultFont', 9, 'bold')).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2)

            # Buffer content label
            buffer_label = ttk.Label(buffer_frame, text="Empty", wraplength=400, justify=tk.LEFT,
                                   font=('Courier', 8))
            buffer_label.grid(row=i, column=1, sticky=tk.W, padx=10, pady=2)
            self.buffer_labels[buffer_key] = buffer_label

        # Bottom section - Instruction List
        bottom_frame = ttk.Frame(main_paned)
        main_paned.add(bottom_frame, minsize=150)

        ttk.Label(bottom_frame, text="Instruction List", font=('TkDefaultFont', 11, 'bold')).pack(
            anchor=tk.W, pady=5)

        # Instruction list text box
        self.inst_text = scrolledtext.ScrolledText(bottom_frame, height=8, font=('Courier', 9))
        self.inst_text.pack(fill=tk.BOTH, expand=True, padx=5)

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
            Instruction(Opcode.ST, rs1=3, rs2=0, imm=200),
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

    def _create_full_memory_display(self):
        """Create full memory display for all 256 bytes"""
        # Display all 256 memory addresses in a scrollable grid
        # Show 16 addresses per row for better layout
        bytes_per_row = 16
        displayed_addresses = 0

        # Create header
        ttk.Label(self.mem_display_frame, text="Address:", font=('Courier', 8, 'bold')).grid(
            row=0, column=0, sticky=tk.W, padx=2)
        for col in range(bytes_per_row):
            ttk.Label(self.mem_display_frame, text=f"+{col:2d}", font=('Courier', 8, 'bold')).grid(
                row=0, column=col+1, sticky=tk.W, padx=2)

        # Create all memory entries
        row = 1
        for addr in range(0, 256, bytes_per_row):
            # Base address label (every 16 bytes)
            ttk.Label(self.mem_display_frame, text=f"{addr:3d}:", font=('Courier', 8, 'bold')).grid(
                row=row, column=0, sticky=tk.W, padx=2)

            # Display 16 bytes per row
            for col in range(bytes_per_row):
                byte_addr = addr + col
                if byte_addr < 256:
                    # Create label for this memory location
                    val_label = ttk.Label(self.mem_display_frame, text=f"{self.simulator.memory[byte_addr]:4d}",
                                        font=('Courier', 8), width=5, relief='solid', borderwidth=1)
                    val_label.grid(row=row, column=col+1, sticky=tk.W, padx=1, pady=1)

                    # Store label reference for updates
                    self.mem_labels[byte_addr] = val_label
                    displayed_addresses += 1

            row += 1

        print(f"Created memory display with {displayed_addresses} addresses")

    def _update_display_internal(self):
        """Internal display update with register values"""
        # Update pipeline stage display with register values
        stages = ["IF", "ID", "RD", "ALU", "MEM", "WB"]
        pipeline_regs = [self.simulator.if_id, self.simulator.id_rd,
                        self.simulator.rd_alu, self.simulator.alu_mem, self.simulator.mem_wb]

        for i, stage in enumerate(stages):
            if stage == "WB":
                # Write-back stage
                if self.simulator.mem_wb.instruction and not self.simulator.mem_wb.bubble:
                    # Show instruction and what register is being written
                    instr = self.simulator.mem_wb.instruction
                    if hasattr(instr, 'rd') and instr.rd is not None:
                        text = f"DONE: R{instr.rd}={self.simulator.mem_wb.alu_result}\n{str(instr)[:30]}"
                    else:
                        text = f"DONE\n{str(instr)[:30]}"
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
                    instr = reg.instruction
                    text = str(instr)

                    # Add register values based on stage and instruction type
                    if stage == "IF":
                        # Show PC address
                        text = f"PC={reg.pc}\n{text}"
                    elif stage == "ID":
                        # Show which registers will be read
                        if instr.opcode in [Opcode.ADD, Opcode.SUB]:
                            if instr.rs1 is not None:
                                text += f"\nR{instr.rs1}={self.simulator.registers[instr.rs1]}"
                            if instr.rs2 is not None:
                                text += f"\nR{instr.rs2}={self.simulator.registers[instr.rs2]}"
                        elif instr.opcode == Opcode.LD:
                            if instr.rs1 is not None:
                                text += f"\nR{instr.rs1}={self.simulator.registers[instr.rs1]}"
                        elif instr.opcode == Opcode.ST:
                            if instr.rs2 is not None:
                                text += f"\nR{instr.rs2}={self.simulator.registers[instr.rs2]}"
                    elif stage == "RD" and hasattr(reg, 'read_data1') and hasattr(reg, 'read_data2'):
                        # Show actual values read from registers in RD stage
                        if instr.opcode == Opcode.ST:
                            # For ST: rs1 is value to store, rs2 is base address
                            text += f"\nval=R{instr.rs1}={reg.read_data1}"
                            if instr.rs2 is not None:
                                text += f"\nbase=R{instr.rs2}={reg.read_data2}"
                        elif instr.opcode == Opcode.LD:
                            # For LD: rs1 is base address
                            if instr.rs1 is not None:
                                text += f"\nbase=R{instr.rs1}={reg.read_data1}"
                        elif instr.opcode in [Opcode.ADD, Opcode.SUB]:
                            # For ALU ops: show both source values
                            text += f"\nR{instr.rs1}={reg.read_data1}"
                            text += f"\nR{instr.rs2}={reg.read_data2}"
                    elif stage == "ALU":
                        # Show ALU result or calculated address
                        if hasattr(reg, 'alu_output') and instr.opcode == Opcode.LD:
                            text += f"\naddr={reg.alu_output}"
                        elif hasattr(reg, 'alu_output') and instr.opcode == Opcode.ST:
                            text += f"\naddr={reg.alu_output}"
                        elif hasattr(reg, 'alu_output'):
                            text += f"\nresult={reg.alu_output}"
                    elif stage == "MEM":
                        # Show memory operation
                        if instr.opcode == Opcode.LD and hasattr(reg, 'alu_result'):
                            text += f"\nmem[{reg.alu_output}]={reg.alu_result}"
                        elif instr.opcode == Opcode.ST and hasattr(reg, 'read_data1'):
                            # Get the stored value
                            addr = reg.alu_output if hasattr(reg, 'alu_output') else 0
                            text += f"\nmem[{addr}]={reg.read_data1}"

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

            self.stage_labels[stage].config(text=text, font=('TkDefaultFont', 8))
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
        elif self.simulator.id_rd.instruction:
            self.current_inst_label.config(text=f"Decode: {self.simulator.id_rd.instruction}")
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

        # Update buffer display
        self._update_buffer_display()

        # Update memory display
        self._update_memory_display()

        self.window.update()

    def _update_buffer_display(self):
        """Update buffer contents display"""
        buffers = {
            'if_id': self.simulator.if_id,
            'id_rd': self.simulator.id_rd,
            'rd_alu': self.simulator.rd_alu,
            'alu_mem': self.simulator.alu_mem,
            'mem_wb': self.simulator.mem_wb
        }

        for buffer_key, buffer_reg in buffers.items():
            text = ""
            if buffer_reg.bubble:
                text = "[Bubble]"
            elif buffer_reg.instruction:
                instr = buffer_reg.instruction
                text = str(instr)

                # Add register values for more detail
                if hasattr(buffer_reg, 'read_data1') or hasattr(buffer_reg, 'alu_output'):
                    details = []

                    if buffer_key == 'rd_alu' and hasattr(buffer_reg, 'read_data1') and hasattr(buffer_reg, 'read_data2'):
                        # RD/ALU buffer has register values
                        if instr.opcode == Opcode.ST:
                            # For ST: show value and base
                            details.append(f"val={buffer_reg.read_data1}")
                            details.append(f"base={buffer_reg.read_data2}")
                        elif instr.opcode == Opcode.LD:
                            details.append(f"base={buffer_reg.read_data1}")
                        elif instr.opcode in [Opcode.ADD, Opcode.SUB]:
                            details.append(f"src1={buffer_reg.read_data1}")
                            details.append(f"src2={buffer_reg.read_data2}")

                    elif buffer_key == 'alu_mem' and hasattr(buffer_reg, 'alu_output'):
                        # ALU/MEM buffer has ALU result
                        details.append(f"alu={buffer_reg.alu_output}")

                    elif buffer_key == 'mem_wb' and hasattr(buffer_reg, 'alu_result'):
                        # MEM/WB buffer has memory result
                        details.append(f"mem={buffer_reg.alu_result}")

                    if details:
                        text += " | " + ", ".join(details)

                # Add stall status
                if hasattr(buffer_reg, 'stalled') and buffer_reg.stalled:
                    text += " [STALLED]"
            else:
                text = "Empty"

            self.buffer_labels[buffer_key].config(text=text)

    def _update_memory_display(self):
        """Update memory display with current values (all 256 addresses)"""
        # Update all memory labels
        for addr, label in self.mem_labels.items():
            try:
                # Try to get original value, skip if conversion fails
                original_text = label.cget("text").strip()
                original_val = int(original_text) if original_text else 0
            except (ValueError, AttributeError):
                original_val = 0

            new_val = self.simulator.memory[addr]

            # Update the label text
            label.config(text=f"{new_val:4d}")

            # Highlight if value changed
            if original_val != new_val:
                label.config(background='#FFF59D')  # Light yellow for changes
            else:
                label.config(background='white')  # Reset background

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
                not any([self.simulator.if_id.instruction, self.simulator.id_rd.instruction,
                        self.simulator.rd_alu.instruction, self.simulator.mem_wb.instruction])):
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
        # Recreate memory display with full 256 bytes
        self.mem_labels.clear()
        for widget in self.mem_display_frame.winfo_children():
            widget.destroy()
        self._create_full_memory_display()
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
        Instruction(Opcode.ADD, rd=2, rs1=1, rs2=0),    # Cycle 2: R2 = R1 + 0
        Instruction(Opcode.ADD, rd=3, rs1=2, rs2=1),    # Cycle 3: R3 = R2 + R1
        Instruction(Opcode.SUB, rd=4, rs1=1, rs2=2),    # R4 = R1 - R2
        Instruction(Opcode.ST, rs1=3, rs2=0, imm=200),  # Store R3
        Instruction(Opcode.BEQ, rs1=3, rs2=4, label="LOOP"),  # Branch
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
