//+------------------------------------------------------------------+
//|                                                 mt5_ea.mq5       |
//|              THE QUANT — MetaTrader 5 bar-export connector       |
//|                                                                  |
//|  Attach to ONE chart. The EA appends every COMPLETED bar of the  |
//|  exported symbols to `<SYMBOL>.csv` files in the terminal's      |
//|  MQL5/Files directory (shared with the Rust binary through       |
//|  `mt5_dir` — the installer symlinks /home/quant/mt5/files to it).|
//|                                                                  |
//|  File format (one line per bar, appended, header on creation):   |
//|      time,symbol,open,high,low,close,volume                      |
//|  where `time` is the bar's OPEN time as Unix epoch seconds.      |
//|  The Rust side (src/simfeed.rs Mt5Feed) tails these files and    |
//|  only trades a timestamp once ALL configured symbols reported.   |
//|                                                                  |
//|  Install: copy to <MT5>/MQL5/Experts/, then attach to any chart. |
//+------------------------------------------------------------------+
#property copyright "THE QUANT"
#property link      "https://github.com/elmaxadore/THE-QUANT"
#property version   "1.00"
#property strict

input string          InpSymbols    = "";         // Symbols to export, comma-separated (empty = chart symbol)
input ENUM_TIMEFRAMES InpTimeframe  = PERIOD_M5;  // Bar timeframe to export
input int             InpBackfill   = 0;          // Export N most-recent bars on start (0 = new bars only)

string   g_symbols[];
datetime g_lastBar[];   // last exported bar OPEN time per symbol

//+------------------------------------------------------------------+
//| Parse the symbol list                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   string list = InpSymbols;
   StringTrimLeft(list); StringTrimRight(list);

   if(StringLen(list) == 0)
      list = _Symbol;                       // default: the chart's own symbol

   // split on commas, trim each entry
   string parts[];
   int n = StringSplit(list, ',', parts);
   if(n <= 0)
     {
      Print("mt5_ea: no symbols to export");
      return(INIT_PARAMETERS_INCORRECT);
     }
   ArrayResize(g_symbols, 0);
   for(int i = 0; i < n; i++)
     {
      string s = parts[i];
      StringTrimLeft(s); StringTrimRight(s);
      if(StringLen(s) == 0) continue;
      ArrayResize(g_symbols, ArraySize(g_symbols) + 1);
      g_symbols[ArraySize(g_symbols) - 1] = s;
     }

   ArrayResize(g_lastBar, ArraySize(g_symbols));
   for(int i = 0; i < ArraySize(g_symbols); i++)
      g_lastBar[i] = 0;

   PrintFormat("mt5_ea: exporting %d symbol(s): %s", ArraySize(g_symbols), list);

   // optional backfill of recent history so a fresh install has data
   if(InpBackfill > 0)
     {
      for(int i = 0; i < ArraySize(g_symbols); i++)
         Backfill(g_symbols[i], InpBackfill);
     }

   EventSetTimer(2);                        // poll even without ticks (quiet markets)
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Main loop / lifecycle                                            |
//+------------------------------------------------------------------+
void OnTick()   { ExportNewBars(); }
void OnTimer()  { ExportNewBars(); }
void OnDeinit(const int reason) { EventKillTimer(); }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Export every completed bar newer than the last exported one      |
//+------------------------------------------------------------------+
void ExportNewBars()
  {
   for(int i = 0; i < ArraySize(g_symbols); i++)
     {
      string sym = g_symbols[i];
      MqlRates rates[];
      // bar index 0 is still forming — start at 1 (completed bars only)
      int copied = CopyRates(sym, InpTimeframe, 1, 2, rates);
      if(copied < 1) continue;

      // newest completed bar is the LAST element returned
      MqlRates bar = rates[copied - 1];
      if(bar.time <= g_lastBar[i]) continue;   // already exported

      AppendBar(sym, bar);
      g_lastBar[i] = bar.time;
     }
  }

//+------------------------------------------------------------------+
//| Backfill the most recent `count` completed bars                  |
//+------------------------------------------------------------------+
void Backfill(const string sym, const int count)
  {
   MqlRates rates[];
   int copied = CopyRates(sym, InpTimeframe, 1, count, rates);
   for(int j = 0; j < copied; j++)
      AppendBar(sym, rates[j]);
   if(copied > 0)
      PrintFormat("mt5_ea: backfilled %d %s bars", copied, sym);
  }

//+------------------------------------------------------------------+
//| Append one bar to <mt5 files>/<SYMBOL>.csv (creating the header) |
//+------------------------------------------------------------------+
void AppendBar(const string sym, const MqlRates &bar)
  {
   string fname = sym + ".csv";
   int handle = FileOpen(fname,
                         FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
     {
      PrintFormat("mt5_ea: cannot open %s (err %d)", fname, GetLastError());
      return;
     }

   bool isEmpty = (FileSize(handle) == 0);
   FileSeek(handle, 0, SEEK_END);

   if(isEmpty)
      FileWriteString(handle, "time,symbol,open,high,low,close,volume\r\n");

   string line = StringFormat("%I64d,%s,%s,%s,%s,%s,%s",
                              (long)bar.time,
                              sym,
                              DoubleToString(bar.open,  5),
                              DoubleToString(bar.high,  5),
                              DoubleToString(bar.low,   5),
                              DoubleToString(bar.close, 5),
                              DoubleToString((double)bar.tick_volume, 1));
   FileWriteString(handle, line + "\r\n");
   FileClose(handle);
  }
//+------------------------------------------------------------------+