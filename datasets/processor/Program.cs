using System;
using System.IO;
using System.Text;

namespace processor
{
    class Program
    {
        static void Main(string[] args)
        {
            var file = File.ReadAllLines("../wine.data");
            var content = new StringBuilder();
            foreach(string line in file){
                var target = line[0].ToString();
                content.AppendLine($"{line.Substring(2, line.Length-2)},{target}");
            }
            System.Console.WriteLine(content.ToString());
            File.WriteAllText("../wine_processed.data", content.ToString());
        }
    }
}
